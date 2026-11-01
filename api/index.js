
const DEFAULT_MODEL = "gemini-3.5-flash-lite";
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_CHARS = 900;
const MAX_ALICE_TEXT = 1024;

const TOTAL_RESPONSE_BUDGET_MS = 3900;
const GEMINI_TIMEOUT_MS = 3300;
const SERPER_TIMEOUT_MS = 350;
const STATE_MAX_BYTES = 850;

const SYSTEM_PROMPT = [
  "Ты — полезный голосовой собеседник в навыке Яндекс Алисы, работающий на базе Google Gemini.",
  "Отвечай естественно, по-русски, кратко и так, чтобы ответ было удобно слушать.",
  "Обычно используй 1–3 коротких предложения. Не используй Markdown, таблицы, списки с символами или длинные вступления.",
  "Учитывай переданную историю диалога и краткую память пользователя, но не выдумывай факты о пользователе.",
  "Если вопрос требует актуальных данных и переданы результаты поиска, опирайся на них; не выдумывай отсутствующие факты.",
  "Если не знаешь ответа, честно скажи об этом."
].join(" ");

function fitAliceText(value) {
  const text = String(value ?? "")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (text.length <= MAX_ALICE_TEXT) {
    return text || "Не удалось подготовить ответ.";
  }

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

function isSearchIntent(text) {
  return /\b(latest|today|current|news|weather|price|exchange rate|recent|search|look up)\b|сегодня|сейчас|последн|новост|погод|курс валют|найди|поищи|актуальн|сколько стоит|свежие данные/i.test(text);
}

function safeHistory(history) {
  if (!Array.isArray(history)) return [];

  return history
    .filter(
      item =>
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

function jsonBytes(value) {
  return Buffer.byteLength(JSON.stringify(value), "utf8");
}

function fitHistoryToState(history, maxBytes = STATE_MAX_BYTES) {
  const result = safeHistory(history).map(item => ({
    role: item.role,
    text: item.text.slice(0, 90)
  }));

  while (
    result.length &&
    jsonBytes({ history: result }) > maxBytes
  ) {
    result.shift();
  }

  return result;
}

function compactMemory(history) {
  return {
    history: fitHistoryToState(history, STATE_MAX_BYTES)
  };
}

function aliceResponse(
  text,
  endSession = false,
  history = [],
  memoryUpdate
) {
  const safeSessionHistory = fitHistoryToState(history);

  const result = {
    version: "1.0",
    response: {
      text: fitAliceText(text),
      end_session: endSession
    },
    session_state: {
      history: safeSessionHistory
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

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = Math.max(100, timeoutMs);

  const timer = setTimeout(
    () => controller.abort(),
    timeout
  );

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const data = await response.json().catch(() => ({}));

    return { response, data };
  } catch (error) {
    if (
      controller.signal.aborted ||
      error?.name === "AbortError"
    ) {
      const timeoutError = new Error("REQUEST_TIMEOUT");
      timeoutError.cause = error;
      throw timeoutError;
    }

    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function searchWeb(query, timeoutMs) {
  const key = process.env.SERPER_API_KEY;

  if (!key || timeoutMs < 150) {
    return "";
  }

  try {
    const { response, data } = await fetchJson(
      "https://google.serper.dev/search",
      {
        method: "POST",
        headers: {
          "X-API-KEY": key,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          q: query,
          gl: "ru",
          hl: "ru",
          num: 3
        })
      },
      Math.min(SERPER_TIMEOUT_MS, timeoutMs)
    );

    if (!response.ok) {
      console.warn(
        "Serper request failed with HTTP",
        response.status
      );
      return "";
    }

    const items = (
      Array.isArray(data.organic)
        ? data.organic
        : []
    )
      .slice(0, 3)
      .map((item, index) => {
        const title = String(item.title || "").slice(0, 160);
        const snippet = String(item.snippet || "").slice(0, 260);

        return `${index + 1}. ${title}: ${snippet}`.trim();
      });

    const answerBox =
      data.answerBox?.answer ||
      data.answerBox?.snippet;

    if (answerBox) {
      items.unshift(
        `Краткий ответ поисковика: ${String(answerBox).slice(0, 260)}`
      );
    }

    return items.length ? items.join("\n") : "";
  } catch (error) {
    console.warn(
      "Serper unavailable; continuing without search:",
      error?.message || "unknown error"
    );

    return "";
  }
}

async function askGemini(
  text,
  history,
  searchContext,
  timeoutMs
) {
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

  const prompt = searchContext
    ? `${text}\n\nРезультаты поиска в интернете:\n${searchContext}\nИспользуй их только там, где они относятся к вопросу.`
    : text;

  contents.push({
    role: "user",
    parts: [{ text: prompt }]
  });

  const { response, data } = await fetchJson(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
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
          temperature: 0.5,
          maxOutputTokens: 180
        }
      })
    },
    Math.min(GEMINI_TIMEOUT_MS, timeoutMs)
  );

  if (!response.ok) {
    const status = Number(response.status);

    const googleMessage = String(
      data?.error?.message || ""
    ).slice(0, 300);

    console.error("Gemini request failed:", {
      status,
      message: googleMessage
    });

    if (status === 400) {
      throw new Error("GEMINI_BAD_REQUEST");
    }

    if (status === 401 || status === 403) {
      throw new Error("GEMINI_KEY_REJECTED");
    }

    if (status === 404) {
      throw new Error("GEMINI_MODEL_NOT_FOUND");
    }

    if (status === 429) {
      throw new Error("GEMINI_QUOTA");
    }

    throw new Error("GEMINI_REQUEST_FAILED");
  }

  const candidate = data?.candidates?.[0];

  const answer = (candidate?.content?.parts || [])
    .map(part =>
      typeof part.text === "string" ? part.text : ""
    )
    .join("")
    .trim();

  if (!answer) {
    if (data?.promptFeedback?.blockReason) {
      throw new Error("GEMINI_BLOCKED");
    }

    throw new Error("GEMINI_EMPTY_RESPONSE");
  }

  return answer;
}

function errorForUser(error) {
  switch (error?.message) {
    case "MISSING_GEMINI_KEY":
      return "Сервер пока не настроен. Проверьте переменную GEMINI_API_KEY в настройках Vercel.";

    case "GEMINI_KEY_REJECTED":
      return "Не удалось подключиться к Gemini. Проверьте API-ключ в настройках Vercel.";

    case "GEMINI_QUOTA":
      return "Сейчас превышен лимит запросов к Gemini. Попробуйте немного позже.";

    case "GEMINI_MODEL_NOT_FOUND":
      return "Указанная модель Gemini не найдена. Проверьте настройку модели на сервере.";

    case "GEMINI_BAD_REQUEST":
      return "Gemini отклонил формат запроса. Попробуйте задать вопрос иначе.";

    case "GEMINI_BLOCKED":
      return "Я не смогла обработать этот запрос. Попробуйте спросить иначе.";

    case "TIME_BUDGET_EXHAUSTED":
    case "REQUEST_TIMEOUT":
      return "Gemini не успела подготовить ответ. Попробуйте задать вопрос ещё раз или короче.";

    default:
      return "Не получилось получить ответ от нейросети. Попробуйте ещё раз чуть позже.";
  }
}

function isClearMemoryCommand(text) {
  return /очисти память|забудь наш разговор|сбрось историю|удали память|очисти историю/i.test(text);
}

export default async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({
      ok: true,
      service: "alice-gemini-serper",
      message: "Webhook is ready. Yandex Dialogs should send POST requests to /api."
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

  const oldHistory = sessionHistory.length
    ? sessionHistory
    : savedMemory;

  if (isClearMemoryCommand(text)) {
    return res.status(200).json(
      aliceResponse(
        "Хорошо, я очистила сохранённую память нашего разговора. Начнём заново!",
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
    let searchContext = "";

    if (
      isSearchIntent(text) &&
      process.env.SERPER_API_KEY
    ) {
      const searchBudget = Math.min(
        SERPER_TIMEOUT_MS,
        Math.max(0, remainingMs() - 3500)
      );

      searchContext = await searchWeb(
        text,
        searchBudget
      );
    }

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
      searchContext,
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
      error?.message || "unknown error"
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
  isSearchIntent,
  safeHistory
};
