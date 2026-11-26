const docsOrigin = "https://basin.rs";
const responseHeaders = { "Cache-Control": "no-store" };
const unavailable = "Ask AI is temporarily unavailable. Please try again.";

function error(message, status, headers = {}) {
    return Response.json(
        { error: message },
        { status, headers: { ...responseHeaders, ...headers } },
    );
}

function sourceLinks(chunks) {
    const sources = new Map();
    for (const { item } of chunks) {
        const url = URL.parse(item.key, docsOrigin);
        if (url?.origin !== docsOrigin) continue;
        const title = item.metadata?.title;
        sources.set(
            url.href,
            typeof title === "string" && title.trim() ? title : url.pathname,
        );
    }
    return Array.from(sources, ([url, title]) => ({ url, title }));
}

async function* answerEvents(stream) {
    let buffer = "";
    let foundSources = false;
    const encoder = new TextEncoder();
    const event = (value) => encoder.encode(`${JSON.stringify(value)}\n`);
    for await (const text of stream.pipeThrough(new TextDecoderStream())) {
        buffer += text;
        let boundary = /\r?\n\r?\n/.exec(buffer);
        while (boundary) {
            const lines = buffer.slice(0, boundary.index).split(/\r?\n/);
            buffer = buffer.slice(boundary.index + boundary[0].length);
            const data = lines
                .filter((line) => line.startsWith("data:"))
                .map((line) => line.slice(5).trimStart())
                .join("\n");
            if (data === "[DONE]") {
                yield event({ done: true });
                return;
            }
            if (data) {
                const payload = JSON.parse(data);
                if (lines.includes("event: chunks")) {
                    const sources = sourceLinks(payload);
                    if (!sources.length) {
                        yield event({
                            text: "I couldn't find an answer in the Basin docs.",
                        });
                        yield event({ done: true });
                        return;
                    }
                    foundSources = true;
                    yield event({ sources });
                } else if (
                    foundSources &&
                    payload.choices?.[0]?.delta?.content
                ) {
                    yield event({ text: payload.choices[0].delta.content });
                }
            }
            boundary = /\r?\n\r?\n/.exec(buffer);
        }
    }
}

export async function askAi(request, env) {
    if (request.method !== "POST") {
        return error("Use POST to submit a question.", 405, { Allow: "POST" });
    }
    if (
        request.headers.get("Content-Type")?.split(";")[0] !==
        "application/json"
    ) {
        return error("Send your question as JSON.", 415);
    }

    const body = await request.json().catch(() => null);
    const question =
        typeof body?.question === "string" ? body.question.trim() : "";
    if (!question || question.length > 500) {
        return error("Enter a question of 1 to 500 characters.", 400);
    }
    const history = body.history === undefined ? [] : body.history;
    if (
        !Array.isArray(history) ||
        history.length > 3 ||
        history.some(
            (turn) =>
                typeof turn?.question !== "string" ||
                !turn.question.trim() ||
                turn.question.length > 500 ||
                typeof turn.answer !== "string" ||
                !turn.answer.trim() ||
                turn.answer.length > 4000,
        )
    ) {
        return error("Invalid conversation history.", 400);
    }

    try {
        const { success } = await env.ASK_RATE_LIMIT.limit({
            key: request.headers.get("CF-Connecting-IP") ?? "local",
        });
        if (!success) {
            return error(
                "Too many questions. Please try again in a minute.",
                429,
            );
        }

        const stream = await env.BASIN_DOCS.chatCompletions({
            stream: true,
            ...(history.length && {
                ai_search_options: { query_rewrite: { enabled: true } },
            }),
            messages: [
                {
                    role: "system",
                    content:
                        "Answer questions about the Basin Rust optimization library using only the retrieved Basin documentation. Use concise Markdown. For code examples, quote an existing example or a contiguous excerpt verbatim in a fenced code block with a language tag. If there is no matching example, explain only what the docs specify and say a matching code example was not found. Never assemble a new example, infer missing API calls, or invent links.",
                },
                ...history.flatMap((turn) => [
                    { role: "user", content: turn.question },
                    { role: "assistant", content: turn.answer },
                ]),
                { role: "user", content: question },
            ],
        });
        return new Response(ReadableStream.from(answerEvents(stream)), {
            headers: {
                ...responseHeaders,
                "Content-Type": "application/x-ndjson",
            },
        });
    } catch (cause) {
        console.error("Ask AI request failed", cause);
        return error(unavailable, 502);
    }
}
