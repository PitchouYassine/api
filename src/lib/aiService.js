// Client Service Layer for AI & YouTube Pipeline APIs
// Uses Google Gemini API with automatic Multi-Key Load Balancing, Parallel Execution & Instant Failover.

const GEMINI_KEYS_STORAGE_KEY = "uyns_gemini_api_keys_v1";

export function getStoredGeminiKeys() {
  try {
    const raw = localStorage.getItem(GEMINI_KEYS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.map((k) => String(k || "").trim()).filter((k) => k.length >= 15);
    }
  } catch {}
  return [];
}

export function setStoredGeminiKeys(keys) {
  try {
    const clean = (Array.isArray(keys) ? keys : [])
      .map((k) => String(k || "").trim())
      .filter((k) => k.length >= 15);
    localStorage.setItem(GEMINI_KEYS_STORAGE_KEY, JSON.stringify(Array.from(new Set(clean))));
  } catch {}
}

const EXTERNAL_CONFIG_STORAGE_KEY = "uyns_external_api_config_v1";

export function getStoredExternalConfig() {
  try {
    const raw = localStorage.getItem(EXTERNAL_CONFIG_STORAGE_KEY);
    if (!raw) {
      return {
        enabled: false,
        apiKey: "",
        baseUrl: "https://api.openai.com/v1",
        model: "gpt-4o-mini",
      };
    }
    const parsed = JSON.parse(raw);
    return {
      enabled: !!parsed?.enabled,
      apiKey: String(parsed?.apiKey || "").trim(),
      baseUrl: String(parsed?.baseUrl || "https://api.openai.com/v1").trim(),
      model: String(parsed?.model || "gpt-4o-mini").trim(),
    };
  } catch {}
  return {
    enabled: false,
    apiKey: "",
    baseUrl: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
  };
}

export function setStoredExternalConfig(cfg) {
  try {
    localStorage.setItem(EXTERNAL_CONFIG_STORAGE_KEY, JSON.stringify(cfg || {}));
  } catch {}
}

async function postJson(endpoint, payload = {}) {
  const storedKeys = getStoredGeminiKeys();
  const extCfg = getStoredExternalConfig();
  const activeProvider =
    payload?.activeProvider ||
    payload?.provider ||
    (extCfg?.enabled ? "external" : "gemini");

  const bodyPayload = {
    ...payload,
    activeProvider,
    provider: activeProvider,
    externalApiConfig: payload?.externalApiConfig || extCfg,
    geminiApiKeys:
      Array.isArray(payload?.geminiApiKeys) && payload.geminiApiKeys.length > 0
        ? payload.geminiApiKeys
        : storedKeys,
  };

  let res;
  try {
    res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyPayload),
    });
  } catch (err) {
    throw new Error(`Network error reaching ${endpoint}: ${err.message || String(err)}`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 413) {
      throw new Error("حجم الطلب كبير جداً (Status 413 - Payload Too Large). تم تحسين وإزالة الوسائط غير الضرورية، يرجى إعادة المحاولة.");
    }
    throw new Error(data.error || `Request to ${endpoint} failed with status ${res.status}`);
  }
  return data;
}

export async function fetchProviderStatus() {
  try {
    const res = await fetch("/api/providers/status");
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function updateServerGeminiConfig(configPayload) {
  if (Array.isArray(configPayload?.apiKeys)) {
    setStoredGeminiKeys(configPayload.apiKeys);
  }
  return postJson("/api/providers/gemini-config", {
    ...configPayload,
    geminiApiKeys: configPayload?.apiKeys || getStoredGeminiKeys(),
  });
}

export async function updateServerExternalConfig(cfg) {
  if (cfg) {
    setStoredExternalConfig(cfg);
  }
  return postJson("/api/providers/external-config", cfg);
}

export async function testExternalApiConnection(cfg) {
  return postJson("/api/providers/external-test", cfg);
}

// Backward-compatible alias
export const updateServerCodeCraftConfig = updateServerGeminiConfig;

export async function fetchYouTubeMetadata(url) {
  return postJson("/api/youtube/metadata", { url });
}

// Helper to strip heavy base64 media from channel profiles before sending to text-only API endpoints.
// This prevents HTTP 413 (Payload Too Large) on Vercel, Nginx, and cloud proxies by reducing payloads from megabytes to ~5-10 KB.
export function stripMediaFromChannel(channel) {
  if (!channel || typeof channel !== "object") return channel;
  const {
    styleReferenceImages,
    thumbnailReferenceImages,
    avatarImages,
    ...clean
  } = channel;
  return clean;
}

// For endpoints that optionally inspect visual references (e.g. image prompts or SEO thumbnails),
// include at most 1 compact reference and ensure oversized base64 strings are omitted.
export function sanitizeChannelForVision(channel, maxImages = 1) {
  if (!channel || typeof channel !== "object") return channel;
  const clean = stripMediaFromChannel(channel);
  const filterCompactImages = (arr) => {
    if (!Array.isArray(arr)) return [];
    return arr
      .slice(0, maxImages)
      .filter((img) => img && typeof img.dataUrl === "string")
      .map((img) => ({
        id: img.id,
        name: img.name,
        // Only include if reasonably compact (< 350 KB) to prevent gateway 413 errors
        dataUrl: img.dataUrl.length < 350_000 ? img.dataUrl : "",
        mimeType: img.mimeType || "image/jpeg",
      }))
      .filter((img) => Boolean(img.dataUrl));
  };

  return {
    ...clean,
    ...(Array.isArray(channel.styleReferenceImages)
      ? { styleReferenceImages: filterCompactImages(channel.styleReferenceImages) }
      : {}),
    ...(Array.isArray(channel.avatarImages)
      ? { avatarImages: filterCompactImages(channel.avatarImages) }
      : {}),
    ...(Array.isArray(channel.thumbnailReferenceImages)
      ? { thumbnailReferenceImages: filterCompactImages(channel.thumbnailReferenceImages) }
      : {}),
  };
}

export async function analyzeReferenceAndConcepts(payload) {
  // Ensure reference thumbnail is not larger than 500KB in payload
  let safePayload = { ...payload };
  if (
    typeof safePayload.thumbnailBase64 === "string" &&
    safePayload.thumbnailBase64.length > 600_000
  ) {
    // If oversized, drop base64 to prevent 413 on strict edge gateways
    safePayload = { ...safePayload, thumbnailBase64: null };
  }
  return postJson("/api/ai/analyze-reference", safePayload);
}

export async function regenerateConcepts(analysis, customDirection = "", provider, codecraftConfig) {
  return postJson("/api/ai/regenerate-concepts", {
    analysis,
    customDirection,
    provider,
    codecraftConfig,
  });
}

export async function generateOrEditScript(payload) {
  // CRITICAL FIX FOR HTTP 413: Strip all heavy base64 images from channelProfile
  // Text script generation only requires promptRules, niche, narrationStyle, etc.
  const cleanPayload = {
    ...payload,
    channelProfile: stripMediaFromChannel(payload?.channelProfile),
  };
  return postJson("/api/ai/generate-script", cleanPayload);
}

export async function generateVoiceoverTTS(text, voiceName = "Charon") {
  return postJson("/api/ai/generate-tts", { text, voiceName });
}

export async function generateSceneImage({ prompt, aspectRatio = "16:9" }) {
  return postJson("/api/ai/generate-scene-image", { prompt, aspectRatio });
}

export async function generateSceneBreakdown(payload) {
  // Strip heavy base64 images from channelProfile — scene breakdown is text-only
  const cleanPayload = {
    ...payload,
    channelProfile: stripMediaFromChannel(payload?.channelProfile),
  };
  return postJson("/api/ai/scene-breakdown", cleanPayload);
}

// High-speed parallel batching across all active Gemini API keys
export async function generateImagePromptsBatch(payload) {
  const { scenes = [], batchSize, ...rest } = payload;
  const cleanRest = {
    ...rest,
    channelProfile: sanitizeChannelForVision(rest?.channelProfile, 1),
  };
  const storedKeys = getStoredGeminiKeys();
  const numWorkers = Math.max(1, storedKeys.length);
  // Auto-tune chunk size based on available keys so all keys work simultaneously
  // e.g. 3 keys & 24 scenes -> 3 chunks of 8 scenes (runs 3x faster with zero queuing)
  const optimalBatchSize = Math.max(4, Math.min(10, Math.ceil(scenes.length / numWorkers)));
  const effectiveBatch = Math.min(batchSize || optimalBatchSize, optimalBatchSize);

  if (scenes.length <= effectiveBatch) {
    const res = await postJson("/api/ai/image-prompts", {
      scenes,
      parallelWorkerIdx: 0,
      ...cleanRest,
    });
    return { prompts: Array.isArray(res.prompts) ? res.prompts : [] };
  }
  const chunks = [];
  for (let i = 0; i < scenes.length; i += effectiveBatch) {
    chunks.push(scenes.slice(i, i + effectiveBatch));
  }
  // Execute chunks in parallel across the Gemini key pool for maximum speed
  const results = await Promise.all(
    chunks.map((chunk, idx) =>
      postJson("/api/ai/image-prompts", {
        scenes: chunk,
        parallelWorkerIdx: idx,
        ...cleanRest,
      })
    )
  );
  const allPrompts = results.flatMap((r) => (Array.isArray(r.prompts) ? r.prompts : []));
  return { prompts: allPrompts };
}

export async function generateMotionPromptsBatch(payload) {
  const { scenes = [], imagePrompts = [], batchSize, ...rest } = payload;
  const cleanRest = {
    ...rest,
    channelProfile: stripMediaFromChannel(rest?.channelProfile),
  };
  const storedKeys = getStoredGeminiKeys();
  const numWorkers = Math.max(1, storedKeys.length);
  const optimalBatchSize = Math.max(5, Math.min(12, Math.ceil(scenes.length / numWorkers)));
  const effectiveBatch = Math.min(batchSize || optimalBatchSize, optimalBatchSize);

  if (scenes.length <= effectiveBatch) {
    const res = await postJson("/api/ai/motion-prompts", {
      scenes,
      imagePrompts,
      parallelWorkerIdx: 0,
      ...cleanRest,
    });
    return { motionPrompts: Array.isArray(res.motionPrompts) ? res.motionPrompts : [] };
  }
  const chunks = [];
  for (let i = 0; i < scenes.length; i += effectiveBatch) {
    chunks.push(scenes.slice(i, i + effectiveBatch));
  }
  const results = await Promise.all(
    chunks.map((chunk, idx) =>
      postJson("/api/ai/motion-prompts", {
        scenes: chunk,
        imagePrompts,
        parallelWorkerIdx: idx,
        ...cleanRest,
      })
    )
  );
  const allMotion = results.flatMap((r) => (Array.isArray(r.motionPrompts) ? r.motionPrompts : []));
  return { motionPrompts: allMotion };
}

export async function generateSeoPackage(payload) {
  const cleanPayload = {
    ...payload,
    channelProfile: sanitizeChannelForVision(payload?.channelProfile, 1),
  };
  return postJson("/api/ai/seo-package", cleanPayload);
}
