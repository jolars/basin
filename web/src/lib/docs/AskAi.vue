<script setup lang="ts">
import MarkdownIt from "markdown-it";
import Sparkles from "~icons/lucide/sparkles";
import CloseIcon from "~icons/lucide/x";
import { computed, reactive, ref, watchEffect } from "vue";

type Source = { url: string; title: string };
type AnswerEvent = { text?: string; sources?: Source[]; done?: boolean };
type Turn = {
    question: string;
    answer: string;
    sources: Source[];
    pending: boolean;
    error: string;
};

const question = ref("");
const turns = ref<Turn[]>([]);
const pending = computed(() => turns.value.at(-1)?.pending ?? false);
const dialog = ref<HTMLDialogElement>();
const conversation = ref<HTMLElement>();
const input = ref<HTMLTextAreaElement>();
const followLatest = ref(true);
const markdown = new MarkdownIt({ html: false }).disable([
    "image",
    "link",
    "autolink",
]);
const renderFence = markdown.renderer.rules.fence;
if (renderFence) {
    markdown.renderer.rules.fence = (...args) => {
        const [tokens, index] = args;
        const language =
            /^[\w+-]+/.exec(tokens[index].info.trim())?.[0] ?? "text";
        return (
            `<div class="language-${language}"><button type="button" class="copy" title="Copy Code" aria-label="Copy code"></button><span class="lang">${language}</span>` +
            renderFence(...args) +
            "</div>"
        );
    };
}
function status(turn: Turn) {
    if (turn.error) return turn.error;
    if (!turn.pending) return "";
    if (!turn.sources.length) return "Searching the Basin docs…";
    const count = turn.sources.length;
    return `Found ${count} source${count === 1 ? "" : "s"}. Writing the answer…`;
}

function trackScroll() {
    const region = conversation.value;
    if (!region) return;
    followLatest.value =
        region.scrollHeight - region.scrollTop - region.clientHeight < 40;
}

watchEffect(
    () => {
        const latest = turns.value.at(-1);
        if (!latest) return;
        // Follow streamed changes unless the reader has scrolled back.
        const { answer, sources, pending, error } = latest;
        if (!(answer || sources.length || pending || error)) return;
        const region = conversation.value;
        if (region && followLatest.value)
            region.scrollTop = region.scrollHeight;
    },
    { flush: "post" },
);

function newConversation() {
    turns.value = [];
    question.value = "";
    input.value?.focus();
}

async function ask() {
    const prompt = question.value.trim();
    if (!prompt || pending.value) return;

    const history = turns.value
        .filter((turn) => !turn.error)
        .slice(-3)
        .map((turn) => ({
            question: turn.question,
            answer: turn.answer.trim().slice(0, 4000),
        }));
    const turn = reactive<Turn>({
        question: prompt,
        answer: "",
        sources: [],
        pending: true,
        error: "",
    });
    followLatest.value = true;
    turns.value.push(turn);
    question.value = "";
    input.value?.focus();
    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: prompt, history }),
            signal: AbortSignal.timeout(60_000),
        });
        if (!response.ok) {
            turn.error = (await response.json()).error;
            return;
        }
        const reader = response.body
            ?.pipeThrough(new TextDecoderStream())
            .getReader();
        if (!reader) throw new Error("Empty response");
        let buffer = "";
        let complete = false;
        try {
            while (!complete) {
                const { done, value } = await reader.read();
                if (done) break;
                const lines = (buffer + value).split("\n");
                buffer = lines.pop() ?? "";
                for (const line of lines) {
                    const event: AnswerEvent = JSON.parse(line);
                    if (event.sources) turn.sources = event.sources;
                    if (event.text) turn.answer += event.text;
                    if (event.done) complete = true;
                }
            }
        } finally {
            await reader.cancel().catch(() => {});
        }
        if (!complete || !turn.answer.trim())
            throw new Error("Incomplete answer");
    } catch {
        turn.error = turn.answer
            ? "The answer was interrupted. Please try again."
            : "Ask AI is temporarily unavailable. Please try again.";
    } finally {
        turn.pending = false;
    }
}
</script>

<template>
    <button
        type="button"
        class="DocSearch DocSearch-Button ask-trigger"
        aria-label="Ask AI"
        aria-haspopup="dialog"
        @click="dialog?.showModal()"
    >
        <span class="DocSearch-Button-Container">
            <Sparkles class="DocSearch-Search-Icon" aria-hidden="true" />
            <span class="DocSearch-Button-Placeholder">Ask AI</span>
        </span>
    </button>
    <dialog ref="dialog" class="ask-ai" aria-labelledby="ask-ai-title">
        <header>
            <h2 id="ask-ai-title">Ask AI</h2>
            <div class="header-actions">
                <button
                    v-if="turns.length"
                    type="button"
                    class="new-conversation"
                    :disabled="pending"
                    @click="newConversation"
                >
                    New conversation
                </button>
                <button
                    type="button"
                    class="close"
                    aria-label="Close Ask AI"
                    @click="dialog?.close()"
                >
                    <CloseIcon aria-hidden="true" />
                </button>
            </div>
        </header>
        <div
            v-if="turns.length"
            ref="conversation"
            class="conversation"
            role="region"
            aria-label="Conversation"
            tabindex="0"
            @scroll.passive="trackScroll"
        >
            <section
                v-for="(turn, index) in turns"
                :key="index"
                class="response"
            >
                <h3 class="submitted-question">{{ turn.question }}</h3>
                <p v-if="status(turn)" class="status" role="status">
                    {{ status(turn) }}
                </p>
                <div
                    v-if="turn.answer"
                    class="answer vp-doc"
                    v-html="markdown.render(turn.answer)"
                ></div>
                <template v-if="!turn.pending && turn.sources.length">
                    <h4>Sources</h4>
                    <ul>
                        <li v-for="source in turn.sources" :key="source.url">
                            <a :href="source.url">{{ source.title }}</a>
                        </li>
                    </ul>
                </template>
            </section>
        </div>
        <form class="question-form" @submit.prevent="ask">
            <label for="basin-question">{{
                turns.length
                    ? "Ask a follow-up"
                    : "Ask a question about the Basin docs"
            }}</label>
            <textarea
                id="basin-question"
                ref="input"
                v-model="question"
                maxlength="500"
                rows="3"
                autofocus
                required
                :placeholder="
                    turns.length
                        ? undefined
                        : 'Which solvers handle box bounds without gradients?'
                "
                @keydown.ctrl.enter.exact.prevent="ask"
            />
            <button
                class="submit"
                type="submit"
                aria-keyshortcuts="Control+Enter"
                :disabled="pending || !question.trim()"
            >
                Ask
            </button>
        </form>
    </dialog>
</template>

<style scoped>
.ask-trigger {
    flex: none;
    white-space: nowrap;
}

.ask-trigger .DocSearch-Search-Icon {
    fill: none;
}

.ask-trigger .DocSearch-Button-Placeholder {
    padding-right: 0;
}

:global(.content-body:has(> .ask-trigger)) {
    reading-flow: flex-visual;
}

:global(.ask-trigger ~ .VPNavBarSearch) {
    order: -1;
    flex-grow: 0;
}

@media (min-width: 768px) {
    .ask-trigger {
        margin-left: 8px;
        margin-right: auto;
        width: max-content;
    }
}

.ask-ai {
    margin: auto;
    width: min(42rem, calc(100% - 2rem));
    max-height: calc(100dvh - 2rem);
    border: 1px solid var(--vp-c-divider);
    border-radius: 0.75rem;
    padding: 1.25rem;
    background-color: var(--vp-c-bg-soft);
    color: var(--vp-c-text-1);
    white-space: normal;
    overflow: hidden;
}

.ask-ai[open] {
    display: flex;
    flex-direction: column;
}

.ask-ai::backdrop {
    background: rgb(0 0 0 / 45%);
}

header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex: none;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.new-conversation {
    color: var(--vp-c-brand-1);
    font-size: 0.875rem;
}

.new-conversation:disabled {
    opacity: 0.55;
}

h2 {
    font-size: 1.25rem;
    font-weight: 700;
}

.close {
    display: grid;
    width: 36px;
    height: 36px;
    place-items: center;
    border-radius: 0.5rem;
}

.close:hover {
    background: var(--vp-c-default-soft);
}

.question-form {
    display: grid;
    flex: none;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

textarea {
    resize: vertical;
    max-height: 24dvh;
    border: 1px solid var(--vp-c-divider);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    background-color: var(--vp-c-bg);
    font-family: inherit;
}

.submit {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    justify-self: start;
    width: max-content;
    min-height: 36px;
    border-radius: 1rem;
    padding: 0.25rem 1rem;
    background-color: var(--vp-button-brand-bg);
    color: var(--vp-button-brand-text);
    font-weight: 600;
}

.submit:disabled {
    opacity: 0.55;
}

.conversation {
    display: grid;
    gap: 0.75rem;
    overflow-y: auto;
    overscroll-behavior: contain;
    min-height: 0;
    margin-top: 0.75rem;
}

.conversation + .question-form {
    border-top: 1px solid var(--vp-c-divider);
    padding-top: 0.75rem;
}

.response {
    min-width: 0;
    border: 1px solid var(--vp-c-divider);
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: var(--vp-c-bg);
    overflow-wrap: anywhere;
}

.submitted-question {
    font-size: 1.125rem;
    font-weight: 650;
}

.status {
    margin-top: 0.5rem;
    color: var(--vp-c-text-2);
    font-size: 0.875rem;
}

.answer {
    margin: 0.75rem 0 1rem;
}

.answer :deep(> :first-child) {
    margin-top: 0;
}

.answer :deep(div[class*="language-"]) {
    margin-inline: 0;
    border-radius: 8px;
}

.response h4 {
    font-weight: 600;
}

ul {
    padding-left: 1.25rem;
    list-style: disc;
}

a {
    color: var(--vp-c-brand-1);
    text-decoration: underline;
}
</style>
