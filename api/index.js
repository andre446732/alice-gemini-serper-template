
const DEFAULT_MODEL = "gemini-3.5-flash-lite";

const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_CHARS = 900;
const MAX_ALICE_TEXT = 1024;
const STATE_MAX_BYTES = 850;

const TOTAL_RESPONSE_BUDGET_MS = 4200;
const GEMINI_TIMEOUT_MS = 3800;

const SYSTEM_PROMPT = [
  "Ты — полезный голосовой собеседник в навыке Яндекс Алисы на базе Google Gemini.",
  "Отвечай естественно, по-русски, кратко и так, чтобы ответ было удобно слушать.",
  "Обычно используй 1–3 коротких предложения.",
  "Не используй Markdown, таблицы и длинные вступления.",
  "Внимательно используй всю переданную историю диалога.",
  "Если пользователь ранее назвал своё имя, профессию, занятие, интересы или сообщил другие личные факты, учитывай их в последующих ответах.",
  "Не проси повторно сообщать сведения, которые уже есть в истории.",
  "Если пользователь спрашивает, что ты о нём помнишь, отвечай на основании истории.",
  "Не выдумывай факты о пользователе.",
  "Если не знаешь ответа, честно скажи об этом."
].join(" ");

function fitAliceText(value) {
  const text = String(value ?? "")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (!text) return "Не удалось подготовить ответ.";

  if (text.length <= MAX_ALICE_TEXT) return text;

  const candidate = text.slice(0, MAX_ALICE_TEXT - 1);

  const sentenceEnd = Math.max(
    candidate.lastIndexOf(". "),
    candidate.lastIndexOf("! "),
    candidate.lastIndexOf("? ")
  );

  if (sentenceEnd >= 300) {
    return candidate.slice(0, sentenceEnd + 1).trim();
  }

  const lastSpace = candidate.lastIndexOf(" ");

  return (
    lastSpace > 200
      ? candidate.slice(0, lastSpace)
      : candidate
  ).trim() + "…";
}

function safeHistory(history) {
  if (!Array.isArray(history)) return [];

  return history
    .filter(item =>
      item &&
      (item.role === "user" || item.role === "model") &&
      typeof item.text === "string"
    )
    .slice(-MAX_HISTORY_MESSAGES)
    .map(item => ({
      role: item.role,
      text: item.text.slice(0, MAX_HISTORY_MESSAGE_CHARS)
    }));
}

// Объединяет старую память и историю текущей сессии.
// Учитывает совпадающие сообщения на границе двух массивов.
function mergeHistory(memoryHistory, sessionHistory) {
  const memory = safeHistory(memoryHistory);
  const session = safeHistory(sessionHistory);

  if (!memory.length) return session;
  if (!session.length) return memory;

  const same = (a, b) =>
    a.role === b.role && a.text === b.text;

  // Если одна история уже содержит другую целиком,
  // возвращаем более полную.
  const containsSequence = (source, sequence) => {
    if (sequence.length > source.length) return false;

    for (let i = 0; i <= source.length - sequence.length; i++) {
      if (sequence.every((item, j) => same(item, source[i + j]))) {
        return true;
      }
    }

    return false;
  };

  if (containsSequence(memory, session)) return memory;
  if (containsSequence(session, memory)) return session;

  // Ищем пересечение конца старой памяти
  // с началом новой сессии.
  let overlap = 0;

  const maxOverlap = Math.min(memory.length, session.length);

  for (let size = 1; size <= maxOverlap; size++) {
    const left = memory.slice(-size);
    const right = session.slice(0, size);

    if (left.every((item, i) => same(item, right[i]))) {
      overlap = size;
    }
  }

  const merged = [
    ...memory,
    ...session.slice(overlap)
  ];

  // Удаляем точные дубли, сохраняя порядок.
  const unique = [];

  for (const item of merged) {
    const previous = unique[unique.length - 1];

    if (!previous || !same(previous, item)) {
      unique.push(item);
    }
  }

  return unique.slice(-MAX_HISTORY_MESSAGES);
}

function jsonBytes(value) {
  return Buffer.byteLength(JSON.stringify(value), "utf8");
}

function fitHistoryToState(history) {
  const result = safeHistory(history).map(item => ({
    role: item.role,
    text: item.text.slice(0, 90)
  }));

  while (
    result.length &&
    jsonBytes({ history: result }) > STATE_MAX_BYTES
  ) {
    result.shift();
  }

  return result;
}

function compactMemory(history) {
  return {
    history: fitHistoryToState(history)
  };
}

function aliceResponse(
  text,
  endSession = false,
  history = [],
  memoryUpdate
) {
  const result = {
    version: "1.0",
    response: {
      text: fitAliceText(text),
      end_session: endSession
    },
    session_state: {
      history: fitHistoryToState(history)
    }
  };

  if (memoryUpdate !== undefined) {
    result.user_state_update = {
      memory: memoryUpdate === null
        ? null
        : compactMemory(memoryUpdate)
    };
  }

  return result;
}

function extractText(data) {
  return (data?.candidates?.[0]?.content?.parts || [])
    .map(part =>
      typeof part.text === "string" ? part.text : ""
    )
    .join("");
}

function firstSentence(text) {
  const match = text.match(/[.!?…](?:["»”')\]]*)?(?:\s|$)/);

  if (!match || match.index === undefined) return "";

  const end = match.index + match[0].length;

  return text.slice(0, end).trim();
}

async function askGemini(text, history, timeoutMs) {
  const startedAt = Date.now();
  const key = process.env.GEMINI_API_KEY;

  if (!key) {
    throw new Error("MISSING_GEMINI_KEY");
  }

  if (timeoutMs < 300) {
    throw new Error("TIME_BUDGET_EXHAUSTED");
  }

  const model = String(
    process.env.GEMINI_MODEL || DEFAULT_MODEL
  ).trim();

  if (!/^[A-Za-z0-9._-]+$/.test(model)) {
    throw new Error("INVALID_MODEL_NAME");
  }

  const contents = safeHistory(history).map(item => ({
    role: item.role,
    parts: [{ text: item.text }]
  }));

  contents.push({
    role: "user",
    parts: [{ text }]
  });

  const controller = new AbortController();

  const timeout = Math.min(
    GEMINI_TIMEOUT_MS,
    timeoutMs
  );

  const timer = setTimeout(
    () => controller.abort(),
    timeout
  );

  let reader;

  try {
    const url =
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;

    console.info("Gemini stream starting", {
      model,
      timeoutMs: timeout,
      historyMessages: history.length
    });

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": key
      },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{ text: SYSTEM_PROMPT }]
        },
        contents,
        generationConfig: {
          temperature: 0.4,
          maxOutputTokens: 120
        }
      }),
      signal: controller.signal
    });

    console.info("Gemini response headers", {
      status: response.status,
      elapsedMs: Date.now() - startedAt
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");

      console.error("Gemini HTTP error", {
        status: response.status,
        message: errorText.slice(0, 400)
      });

      if (response.status === 400) {
        throw new Error("GEMINI_BAD_REQUEST");
      }

      if (response.status === 401 || response.status === 403) {
        throw new Error("GEMINI_KEY_REJECTED");
      }

      if (response.status === 404) {
        throw new Error("GEMINI_MODEL_NOT_FOUND");
      }

      if (response.status === 429) {
        throw new Error("GEMINI_QUOTA");
      }

      throw new Error("GEMINI_REQUEST_FAILED");
    }

    if (!response.body) {
      throw new Error("GEMINI_EMPTY_RESPONSE");
    }

    reader = response.body.getReader();

    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let firstTokenLogged = false;

    async function processEvent(event) {
      const lines = event.split("\n");

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;

        const raw = line.slice(5).trim();

        if (!raw || raw === "[DONE]") continue;

        let data;

        try {
          data = JSON.parse(raw);
        } catch {
          console.warn("Gemini SSE: invalid JSON event");
          continue;
        }

        if (data?.promptFeedback?.blockReason) {
          throw new Error("GEMINI_BLOCKED");
        }

        const fragment = extractText(data);

        if (fragment) {
          answer += fragment;

          if (!firstTokenLogged) {
            firstTokenLogged = true;

            console.info("Gemini first text received", {
              elapsedMs: Date.now() - startedAt
            });
          }
        }
      }
    }

    while (true) {
      const { value, done } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let separator;

      while ((separator = buffer.indexOf("\n\n")) !== -1) {
        const event = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);

        await processEvent(event);

        const sentence = firstSentence(answer);

        if (sentence.length >= 25) {
          console.info("Gemini first sentence ready", {
            elapsedMs: Date.now() - startedAt
          });

          await reader.cancel().catch(() => {});

          return sentence;
        }

        if (answer.length >= 250) {
          await reader.cancel().catch(() => {});
          return answer.trim();
        }
      }
    }

    if (buffer.trim()) {
      await processEvent(buffer);
    }

    const finalAnswer = answer.trim();

    console.info("Gemini stream finished", {
      elapsedMs: Date.now() - startedAt,
      answerLength: finalAnswer.length
    });

    if (!finalAnswer) {
      throw new Error("GEMINI_EMPTY_RESPONSE");
    }

    return finalAnswer;
  } catch (error) {
    if (controller.signal.aborted || error?.name === "AbortError") {
      console.warn("Gemini stream timeout", {
        elapsedMs: Date.now() - startedAt
      });

      throw new Error("REQUEST_TIMEOUT");
    }

    throw error;
  } finally {
    clearTimeout(timer);

    if (reader) {
      await reader.cancel().catch(() => {});
    }
  }
}

function errorForUser(error) {
  switch (error?.message) {
    case "MISSING_GEMINI_KEY":
      return "Сервер пока не настроен. Проверьте GEMINI_API_KEY в Vercel.";

    case "GEMINI_KEY_REJECTED":
      return "Не удалось подключиться к Gemini. Проверьте API-ключ.";

    case "GEMINI_QUOTA":
      return "Сейчас превышен лимит запросов Gemini. Попробуйте позже.";

    case "GEMINI_MODEL_NOT_FOUND":
      return "Модель Gemini не найдена. Проверьте настройку модели.";

    case "GEMINI_BAD_REQUEST":
      return "Gemini отклонил запрос. Попробуйте задать вопрос иначе.";

    case "GEMINI_BLOCKED":
      return "Я не смогла обработать этот запрос. Попробуйте спросить иначе.";

    case "TIME_BUDGET_EXHAUSTED":
    case "REQUEST_TIMEOUT":
      return "Gemini не успела подготовить ответ. Попробуйте ещё раз.";

    default:
      return "Не получилось получить ответ от нейросети. Попробуйте позже.";
  }
}

function isClearMemoryCommand(text) {
  return /очисти память|забудь наш разговор|сбрось историю|удали память|очисти историю/i.test(text);
}

export default async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({
      ok: true,
      service: "alice-gemini",
      message: "Gemini webhook is ready."
    });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");

    return res.status(405).json({
      error: "Method not allowed"
    });
  }

  const startedAt = Date.now();

  const remainingMs = () =>
    TOTAL_RESPONSE_BUDGET_MS - (Date.now() - startedAt);

  const body =
    req.body && typeof req.body === "object"
      ? req.body
      : {};

  const text = String(
    body?.request?.command ||
    body?.request?.original_utterance ||
    ""
  ).trim();

  const sessionHistory = safeHistory(
    body?.state?.session?.history
  );

  const savedMemory = safeHistory(
    body?.state?.user?.memory?.history
  );

  // Главное исправление:
  // используем одновременно историю сессии и память пользователя.
  const oldHistory = mergeHistory(
    savedMemory,
    sessionHistory
  );

  console.info("Alice state received", {
    sessionMessages: sessionHistory.length,
    memoryMessages: savedMemory.length,
    mergedMessages: oldHistory.length,
    userIdPresent: Boolean(body?.session?.user?.user_id)
  });

  if (isClearMemoryCommand(text)) {
    console.info("Alice memory clear requested");

    return res.status(200).json(
      aliceResponse(
        "Хорошо, я очистила сохранённую память. Начнём заново!",
        false,
        [],
        null
      )
    );
  }

  if (!text) {
    return res.status(200).json(
      aliceResponse(
        "Привет! Я Gemini. Задай мне вопрос, и мы начнём разговор.",
        false,
        oldHistory
      )
    );
  }

  try {
    const geminiBudget = Math.max(
      0,
      Math.min(
        GEMINI_TIMEOUT_MS,
        remainingMs() - 250
      )
    );

    const answer = await askGemini(
      text,
      oldHistory,
      geminiBudget
    );

    const history = fitHistoryToState([
      ...oldHistory,
      {
        role: "user",
        text: text.slice(0, MAX_HISTORY_MESSAGE_CHARS)
      },
      {
        role: "model",
        text: answer.slice(0, MAX_HISTORY_MESSAGE_CHARS)
      }
    ]);

    console.info("Alice response ready", {
      elapsedMs: Date.now() - startedAt,
      historyMessages: history.length,
      memoryUpdate: true
    });

    return res.status(200).json(
      aliceResponse(
        answer,
        false,
        history,
        history
      )
    );
  } catch (error) {
    console.warn(
      "Alice webhook fallback:",
      error?.message || "unknown error",
      {
        elapsedMs: Date.now() - startedAt
      }
    );

    return res.status(200).json(
      aliceResponse(
        errorForUser(error),
        false,
        oldHistory
      )
    );
  }
}

export {
  aliceResponse,
  compactMemory,
  fitAliceText,
  fitHistoryToState,
  mergeHistory,
  safeHistory
};
