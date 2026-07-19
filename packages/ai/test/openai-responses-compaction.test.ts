import type { ResponseOutputItem, ResponseStreamEvent } from "openai/resources/responses/responses.js";
import { describe, expect, it, vi } from "vitest";
import {
	convertResponsesMessages,
	getOpenAIResponsesCompactionCount,
	processResponsesStream,
} from "../src/api/openai-responses-shared.ts";
import type { AssistantMessage, Context, Model, ToolResultMessage } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";

const COMPACTION_ITEM_MAX_BYTES = 16 * 1024 * 1024;
const RETAINED_WINDOW_MAX_BYTES = 32 * 1024 * 1024;
const ALLOWED_TOOL_CALL_PROVIDERS = new Set(["compaction-proxy"]);

function model(id = "gpt-test"): Model<"openai-responses"> {
	return {
		id,
		name: id,
		api: "openai-responses",
		provider: "compaction-proxy",
		baseUrl: "https://example.invalid/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 272_000,
		maxTokens: 16_000,
	};
}

function outputFor(selectedModel = model()): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: selectedModel.api,
		provider: selectedModel.provider,
		model: selectedModel.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 1,
	};
}

function compaction(id: string, encryptedContent = `opaque-${id}`): ResponseOutputItem {
	return { type: "compaction", id, encrypted_content: encryptedContent };
}

function functionCall(callId: string): ResponseOutputItem {
	return {
		type: "function_call",
		id: `fc_${callId}`,
		call_id: callId,
		name: "inspect",
		arguments: JSON.stringify({ callId }),
	};
}

function message(id: string, text: string): ResponseOutputItem {
	return {
		type: "message",
		id,
		role: "assistant",
		status: "completed",
		content: [{ type: "output_text", text, annotations: [], logprobs: [] }],
	};
}

function addedShape(item: ResponseOutputItem): ResponseOutputItem {
	if (item.type === "function_call") return { ...item, arguments: "" };
	if (item.type === "message") return { ...item, content: [] };
	return { ...item };
}

async function* successfulEvents(items: readonly ResponseOutputItem[]): AsyncIterable<ResponseStreamEvent> {
	let sequenceNumber = 0;
	for (let outputIndex = 0; outputIndex < items.length; outputIndex++) {
		const item = items[outputIndex];
		yield {
			type: "response.output_item.added",
			sequence_number: sequenceNumber++,
			output_index: outputIndex,
			item: addedShape(item),
		} as ResponseStreamEvent;
		yield {
			type: "response.output_item.done",
			sequence_number: sequenceNumber++,
			output_index: outputIndex,
			item,
		} as ResponseStreamEvent;
	}
	yield {
		type: "response.completed",
		sequence_number: sequenceNumber,
		response: { id: "resp_test", status: "completed", output: [...items] },
	} as ResponseStreamEvent;
}

async function* events(values: readonly ResponseStreamEvent[]): AsyncIterable<ResponseStreamEvent> {
	for (const value of values) yield value;
}

async function retain(items: readonly ResponseOutputItem[], selectedModel = model()) {
	const output = outputFor(selectedModel);
	const stream = new AssistantMessageEventStream();
	const push = vi.spyOn(stream, "push");
	await processResponsesStream(successfulEvents(items), output, stream, selectedModel);
	return { output, emitted: push.mock.calls.map(([event]) => event.type) };
}

function toolResult(callId: string, text = `result-${callId}`): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: `${callId}|fc_${callId}`,
		toolName: "inspect",
		content: [{ type: "text", text }],
		isError: false,
		timestamp: 2,
	};
}

function replay(selectedModel: Model<"openai-responses">, messages: Context["messages"]) {
	return convertResponsesMessages(
		selectedModel,
		{ systemPrompt: "keep policy", messages },
		ALLOWED_TOOL_CALL_PROVIDERS,
	);
}

function itemWithJsonBytes<T extends Record<string, unknown>>(base: T, bytes: number, field: keyof T): T {
	const emptyBytes = new TextEncoder().encode(JSON.stringify(base)).byteLength;
	if (bytes < emptyBytes || base[field] !== "") throw new Error("invalid sized-item fixture");
	return { ...base, [field]: "x".repeat(bytes - emptyBytes) };
}

describe("OpenAI Responses opaque compaction retention", () => {
	it("keeps no-pivot serialization unchanged", async () => {
		const selectedModel = model();
		const nativeMessage = message("msg_normal", "normal reply");
		const { output } = await retain([nativeMessage], selectedModel);
		const input = replay(selectedModel, [
			{ role: "user", content: "initial", timestamp: 0 },
			output,
			{ role: "user", content: "follow up", timestamp: 3 },
		]);

		expect(getOpenAIResponsesCompactionCount(output)).toBe(0);
		expect(input[0]).toEqual({ role: "developer", content: "keep policy" });
		expect(input).toContainEqual({
			type: "message",
			id: "msg_normal",
			role: "assistant",
			status: "completed",
			phase: undefined,
			content: [{ type: "output_text", text: "normal reply", annotations: [] }],
		});
	});

	it("replays one opaque pivot unchanged with its ordered function call and result", async () => {
		const selectedModel = model();
		const before = message("msg_before", "must be pruned");
		const pivot = compaction("cmp_1");
		const call = functionCall("call_1");
		const { output, emitted } = await retain([before, pivot, call], selectedModel);
		const input = replay(selectedModel, [
			{ role: "user", content: "pre-pivot sentinel", timestamp: 0 },
			output,
			toolResult("call_1"),
		]);

		expect(getOpenAIResponsesCompactionCount(output)).toBe(1);
		expect(emitted).toEqual(["text_start", "text_end", "toolcall_start", "toolcall_end"]);
		expect(input).toHaveLength(3);
		expect(input[0]).toBe(pivot);
		expect(input[1]).toBe(call);
		expect(input[2]).toEqual({ type: "function_call_output", call_id: "call_1", output: "result-call_1" });
		expect(JSON.stringify(input)).not.toContain("pre-pivot sentinel");
		expect(JSON.stringify(input)).not.toContain("must be pruned");
		expect(Object.isFrozen(pivot)).toBe(true);
	});

	it("selects the newest pivot across turns and within one response", async () => {
		const selectedModel = model();
		const firstPivot = compaction("cmp_1");
		const firstCall = functionCall("call_1");
		const first = await retain([firstPivot, firstCall], selectedModel);
		const secondPivot = compaction("cmp_2");
		const discardedPivot = compaction("cmp_2b");
		const secondCall = functionCall("call_2");
		const second = await retain([secondPivot, discardedPivot, secondCall], selectedModel);
		const input = replay(selectedModel, [first.output, toolResult("call_1"), second.output, toolResult("call_2")]);

		expect(getOpenAIResponsesCompactionCount(second.output)).toBe(2);
		expect(input).toHaveLength(3);
		expect(input[0]).toBe(discardedPivot);
		expect(input[1]).toBe(secondCall);
		expect(input[2]).toMatchObject({ type: "function_call_output", call_id: "call_2" });
		expect(input).not.toContain(firstPivot);
		expect(input).not.toContain(secondPivot);
	});

	it("keeps WeakMap state private from spread copies and rejects foreign identity", async () => {
		const selectedModel = model();
		const { output } = await retain([compaction("cmp_private")], selectedModel);
		const copied = { ...output };

		expect(getOpenAIResponsesCompactionCount(copied)).toBe(0);
		expect(Object.keys(output)).not.toContain("compaction");
		expect(JSON.stringify(output)).not.toContain("opaque-cmp_private");
		expect(() => replay(model("other-model"), [output])).toThrow("openai_responses_compaction_invalid");

		output.provider = "foreign-provider";
		expect(() => getOpenAIResponsesCompactionCount(output)).toThrow("openai_responses_compaction_invalid");
	});

	it("rejects orphaned, missing, and out-of-order function results", async () => {
		const selectedModel = model();
		const pivotOnly = await retain([compaction("cmp_only")], selectedModel);
		expect(() => replay(selectedModel, [pivotOnly.output, toolResult("orphan")])).toThrow(
			"openai_responses_compaction_invalid",
		);

		const oneCall = await retain([compaction("cmp_call"), functionCall("call_1")], selectedModel);
		expect(() => replay(selectedModel, [oneCall.output])).toThrow("openai_responses_compaction_invalid");

		const parallel = await retain(
			[compaction("cmp_parallel"), functionCall("call_1"), functionCall("call_2")],
			selectedModel,
		);
		expect(() => replay(selectedModel, [parallel.output, toolResult("call_2"), toolResult("call_1")])).toThrow(
			"openai_responses_compaction_invalid",
		);
	});
});

describe("OpenAI Responses compaction stream validation", () => {
	it.each([
		{
			name: "missing output index zero",
			values: [
				{
					type: "response.output_item.added",
					output_index: 1,
					item: compaction("cmp_index"),
				} as ResponseStreamEvent,
			],
		},
		{
			name: "done without added",
			values: [
				{
					type: "response.output_item.done",
					output_index: 0,
					item: compaction("cmp_done"),
				} as ResponseStreamEvent,
			],
		},
		{
			name: "mutated item identity",
			values: [
				{
					type: "response.output_item.added",
					output_index: 0,
					item: compaction("cmp_added"),
				} as ResponseStreamEvent,
				{
					type: "response.output_item.done",
					output_index: 0,
					item: compaction("cmp_changed"),
				} as ResponseStreamEvent,
			],
		},
	])("uses the fixed safe failure for $name", async ({ values }) => {
		await expect(
			processResponsesStream(events(values), outputFor(), new AssistantMessageEventStream(), model()),
		).rejects.toThrow("openai_responses_compaction_invalid");
	});

	it("rejects duplicate and unfinished indexes", async () => {
		const pivot = compaction("cmp_duplicate");
		const duplicate = [
			{ type: "response.output_item.added", output_index: 0, item: addedShape(pivot) },
			{ type: "response.output_item.added", output_index: 0, item: addedShape(pivot) },
		] as ResponseStreamEvent[];
		await expect(
			processResponsesStream(events(duplicate), outputFor(), new AssistantMessageEventStream(), model()),
		).rejects.toThrow("openai_responses_compaction_invalid");

		const unfinished = [
			{ type: "response.output_item.added", output_index: 0, item: addedShape(pivot) },
			{ type: "response.completed", response: { id: "resp", status: "completed" } },
		] as ResponseStreamEvent[];
		await expect(
			processResponsesStream(events(unfinished), outputFor(), new AssistantMessageEventStream(), model()),
		).rejects.toThrow("openai_responses_compaction_invalid");
	});

	it("never includes opaque data in malformed-state errors", async () => {
		const sentinel = "OPAQUE_SENTINEL_MUST_NOT_LEAK";
		const malformed = [
			{ type: "response.output_item.added", output_index: 0, item: compaction("cmp_a", sentinel) },
			{ type: "response.output_item.done", output_index: 0, item: compaction("cmp_b", sentinel) },
		] as ResponseStreamEvent[];

		let failure = "";
		try {
			await processResponsesStream(events(malformed), outputFor(), new AssistantMessageEventStream(), model());
		} catch (error) {
			failure = error instanceof Error ? error.message : String(error);
		}
		expect(failure).toBe("openai_responses_compaction_invalid");
		expect(failure).not.toContain(sentinel);
	});

	it("does not retain compaction state after provider failure or early EOF", async () => {
		const pivot = compaction("cmp_transient");
		const failedOutput = outputFor();
		const failed = [
			{ type: "response.output_item.added", output_index: 0, item: addedShape(pivot) },
			{ type: "response.output_item.done", output_index: 0, item: pivot },
			{
				type: "response.failed",
				response: { id: "resp", status: "failed", error: { code: "server_error", message: "safe" } },
			},
		] as ResponseStreamEvent[];
		await expect(
			processResponsesStream(events(failed), failedOutput, new AssistantMessageEventStream(), model()),
		).rejects.toThrow("server_error: safe");
		expect(getOpenAIResponsesCompactionCount(failedOutput)).toBe(0);

		const eofOutput = outputFor();
		await expect(
			processResponsesStream(events(failed.slice(0, 2)), eofOutput, new AssistantMessageEventStream(), model()),
		).rejects.toThrow("OpenAI Responses stream ended before a terminal response event");
		expect(getOpenAIResponsesCompactionCount(eofOutput)).toBe(0);
	});
});

describe("OpenAI Responses compaction byte limits", () => {
	it("accepts exactly 16 MiB and rejects one byte more without leaking content", { timeout: 30_000 }, async () => {
		const exact = itemWithJsonBytes(
			{ type: "compaction", id: "cmp_exact", encrypted_content: "" },
			COMPACTION_ITEM_MAX_BYTES,
			"encrypted_content",
		) as ResponseOutputItem;
		const accepted = await retain([exact]);
		expect(getOpenAIResponsesCompactionCount(accepted.output)).toBe(1);

		const oversized = itemWithJsonBytes(
			{ type: "compaction", id: "cmp_large", encrypted_content: "" },
			COMPACTION_ITEM_MAX_BYTES + 1,
			"encrypted_content",
		) as ResponseOutputItem;
		await expect(
			processResponsesStream(successfulEvents([oversized]), outputFor(), new AssistantMessageEventStream(), model()),
		).rejects.toThrow("openai_responses_compaction_oversized");
	});

	it(
		"accepts exactly 32 MiB from pivot through native tail and rejects one byte more",
		{ timeout: 30_000 },
		async () => {
			const pivot = itemWithJsonBytes(
				{ type: "compaction", id: "cmp_window", encrypted_content: "" },
				COMPACTION_ITEM_MAX_BYTES,
				"encrypted_content",
			) as ResponseOutputItem;
			const exactTail = itemWithJsonBytes(
				{ type: "file_search_call", id: "fs_exact", padding: "" },
				RETAINED_WINDOW_MAX_BYTES - COMPACTION_ITEM_MAX_BYTES,
				"padding",
			) as unknown as ResponseOutputItem;
			const accepted = await retain([pivot, exactTail]);
			expect(getOpenAIResponsesCompactionCount(accepted.output)).toBe(1);

			const oversizedTail = itemWithJsonBytes(
				{ type: "file_search_call", id: "fs_large", padding: "" },
				RETAINED_WINDOW_MAX_BYTES - COMPACTION_ITEM_MAX_BYTES + 1,
				"padding",
			) as unknown as ResponseOutputItem;
			await expect(
				processResponsesStream(
					successfulEvents([pivot, oversizedTail]),
					outputFor(),
					new AssistantMessageEventStream(),
					model(),
				),
			).rejects.toThrow("openai_responses_compaction_oversized");
		},
	);
});
