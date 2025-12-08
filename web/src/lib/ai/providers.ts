/**
 * LLM Provider Abstraction
 * 
 * Supports Gemini 2.5 Flash (primary) and GPT-4.1 mini (fallback)
 * Uses native fetch with streaming support for edge runtime compatibility
 */

export type LLMProvider = "gemini" | "openai";

/**
 * Find the longest common prefix between two strings
 */
function findCommonPrefix(str1: string, str2: string): string {
  let i = 0;
  const minLen = Math.min(str1.length, str2.length);
  while (i < minLen && str1[i] === str2[i]) {
    i++;
  }
  return str1.substring(0, i);
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface StreamChunk {
  content: string;
  done: boolean;
}

/**
 * Stream response from Gemini API
 * Based on official Gemini API documentation: https://ai.google.dev/api/generate-content
 */
export async function* streamGeminiResponse(
  messages: ChatMessage[],
  systemPrompt: string,
  apiKey: string
): AsyncGenerator<StreamChunk, void, unknown> {
  // Validate inputs
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("Gemini API key is required");
  }

  // Basic API key format validation (Google API keys typically start with "AIza")
  if (!apiKey.startsWith("AIza") && apiKey.length < 20) {
    console.warn("[Gemini] API key format may be invalid. Expected format: AIza...");
  }

  if (!messages || messages.length === 0) {
    throw new Error("Messages array cannot be empty");
  }

  // Ensure we have at least one user message
  const hasUserMessage = messages.some(m => m.role === "user" && m.content.trim());
  if (!hasUserMessage) {
    throw new Error("At least one non-empty user message is required");
  }

  // Convert messages to Gemini format
  // Gemini uses "model" for assistant, "user" for user
  const geminiContents = messages
    .filter(m => m.role !== "system")
    .map(m => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));

  // Build request body - use systemInstruction for system prompt (doesn't count against content tokens)
  const contents = geminiContents;
  
  // Ensure we have at least one content item
  if (contents.length === 0) {
    throw new Error("Request body must contain at least one message");
  }
  
  const requestBody: any = {
    contents,
    generationConfig: {
      temperature: 0.8,
      topK: 40,
      topP: 0.95,
      maxOutputTokens: 8192, // Increased from 1024 to allow longer responses
    },
  };
  
  // Use systemInstruction for system prompt (more efficient, doesn't count as user message)
  if (systemPrompt && systemPrompt.trim()) {
    requestBody.systemInstruction = {
      parts: [{ text: systemPrompt.trim() }],
    };
  }

  // Use gemini-2.5-flash (primary), fallback to gemini-2.5-flash-lite
  const models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"];
  let lastError: Error | null = null;

  for (const model of models) {
    try {
      // Use SSE format with alt=sse parameter and x-goog-api-key header for immediate streaming
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
      
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Gemini API error (${model}): ${response.status}`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage += ` - ${errorJson.error?.message || errorText}`;
        } catch {
          errorMessage += ` - ${errorText}`;
        }
        lastError = new Error(errorMessage);
        console.error(`[Gemini] Model ${model} failed:`, errorMessage);
        // Try next model if this one fails
        if (model !== models[models.length - 1]) continue;
        throw lastError;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body reader");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let hasYieldedContent = false;
      let accumulatedText = ""; // Track accumulated text to extract deltas
      let seenFinishReason = false; // Track if we've seen finishReason, but continue processing

      try {
        // Process SSE stream line-by-line for immediate streaming
        // Format: "data: {...}\n\n" where each line is a JSON object
        // CRITICAL: Continue processing until reader.done, don't return early on finishReason
        while (true) {
          const { done, value } = await reader.read();
          
          // Decode chunk immediately
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;
          }
          
          // Process complete lines immediately (SSE format)
          const lines = buffer.split('\n');
          // Keep the last incomplete line in buffer
          buffer = lines.pop() || "";
          
          // Process each complete line
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue; // Skip empty lines
            
            // SSE format: "data: {...}"
            if (trimmed.startsWith('data: ')) {
              const jsonStr = trimmed.slice(6); // Remove "data: " prefix
              
              try {
                const data = JSON.parse(jsonStr);
                
                // Check for errors first
                if (data.error) {
                  throw new Error(`Gemini API error: ${data.error.message || JSON.stringify(data.error)}`);
                }

                // Check for promptFeedback errors
                if (data.promptFeedback?.blockReason) {
                  throw new Error(`Gemini API blocked content: ${data.promptFeedback.blockReason}`);
                }
                
                // Process candidates
                const candidates = data?.candidates;
                if (candidates && candidates.length > 0) {
                  const candidate = candidates[0];
                  
                  // Track finishReason but DON'T return early - continue processing all chunks
                  if (candidate.finishReason) {
                    if (candidate.finishReason === "SAFETY" || candidate.finishReason === "RECITATION") {
                      throw new Error(`Gemini API safety filter: ${candidate.finishReason}`);
                    }
                    if (!seenFinishReason) {
                      seenFinishReason = true;
                      console.log(`[Gemini Stream] Finish reason detected: ${candidate.finishReason} (continuing to process remaining chunks)`);
                    }
                  }
                  
                  // Always process content parts, even if finishReason is present
                  // This ensures we capture ALL text from ALL chunks
                  if (candidate.content?.parts && candidate.content.parts.length > 0) {
                    for (const part of candidate.content.parts) {
                      const text = part.text || "";
                      if (text) {
                        // SIMPLIFIED LOGIC: Gemini sends FULL accumulated text in each chunk
                        // Always use the longest text we've seen - it contains all previous content
                        if (accumulatedText.length === 0) {
                          // First chunk - no accumulated text yet
                          hasYieldedContent = true;
                          console.log(`[Gemini Stream] First chunk: ${text.length} characters - sending immediately`);
                          // Yield two characters at a time with 25ms delay between chunks
                          for (let i = 0; i < text.length; i += 2) {
                            const chunk = text.slice(i, i + 2);
                            yield { content: chunk, done: false };
                            // 25ms delay between chunks for readable streaming
                            if (i < text.length - 2) {
                              await new Promise(resolve => setTimeout(resolve, 25));
                            }
                          }
                          accumulatedText = text;
                        } else if (text.length > accumulatedText.length) {
                          // New chunk has more text - extract and yield the delta
                          const delta = text.slice(accumulatedText.length);
                          hasYieldedContent = true;
                          console.log(`[Gemini Stream] Chunk ${seenFinishReason ? '[POST-FINISH]' : ''}: Received ${delta.length} new chars (accumulated: ${accumulatedText.length} → ${text.length})`);
                          
                          // Yield two characters at a time with 25ms delay between chunks
                          for (let i = 0; i < delta.length; i += 2) {
                            const chunk = delta.slice(i, i + 2);
                            yield { content: chunk, done: false };
                            // 25ms delay between chunks for readable streaming
                            if (i < delta.length - 2) {
                              await new Promise(resolve => setTimeout(resolve, 25));
                            }
                          }
                          // Update accumulated to the full text
                          accumulatedText = text;
                        } else if (text.length === accumulatedText.length && text === accumulatedText) {
                          // Same text - confirmation chunk (normal, especially with finishReason)
                          if (!seenFinishReason && accumulatedText.length > 0) {
                            console.log(`[Gemini Stream] Received confirmation chunk (text matches: ${text.length} chars)`);
                          }
                        } else if (text.length < accumulatedText.length) {
                          // Shorter text - CRITICAL: This might be a DIFFERENT segment, not accumulated text
                          // Check if it's a continuation or a different part
                          const accumulatedEnd = accumulatedText.substring(Math.max(0, accumulatedText.length - 50));
                          const receivedStart = text.substring(0, Math.min(50, text.length));
                          
                          // If received text doesn't appear in accumulated, it might be a new segment
                          if (!accumulatedText.includes(text) && !text.includes(accumulatedText)) {
                            // Different segment - CONCATENATE it with proper spacing
                            console.warn(`[Gemini Stream] Received different segment (${text.length} chars). Concatenating to accumulated (${accumulatedText.length} chars).`);
                            console.warn(`[Gemini Stream] Accumulated ends: "...${accumulatedEnd}"`);
                            console.warn(`[Gemini Stream] Received starts: "${receivedStart}..."`);

                            // Check if we need a space between segments
                            const needsSpace = accumulatedText.length > 0 &&
                              !accumulatedText.endsWith(' ') &&
                              !accumulatedText.endsWith('\n') &&
                              !text.startsWith(' ') &&
                              !text.startsWith('\n');

                            if (needsSpace) {
                              yield { content: ' ', done: false };
                              accumulatedText += ' ';
                            }

                            // Yield the new segment with 50ms delay between chunks
                            hasYieldedContent = true;
                            for (let i = 0; i < text.length; i += 2) {
                              const chunk = text.slice(i, i + 2);
                              yield { content: chunk, done: false };
                              // 25ms delay between chunks for readable streaming
                              if (i < text.length - 2) {
                                await new Promise(resolve => setTimeout(resolve, 25));
                              }
                            }
                            // Append to accumulated (don't replace)
                            accumulatedText += text;
                          } else {
                            // It's a subset of accumulated - ignore it
                            console.warn(`[Gemini Stream] Received shorter text (${text.length} < ${accumulatedText.length}) that's a subset. Keeping longer accumulated text.`);
                          }
                        } else {
                          // Same length but different content - this is unexpected, log it
                          console.warn(`[Gemini Stream] WARNING: Same length (${text.length}) but different content!`);
                          console.warn(`[Gemini Stream] Accumulated: "${accumulatedText.substring(0, 50)}..."`);
                          console.warn(`[Gemini Stream] Received: "${text.substring(0, 50)}..."`);
                          // Check if one contains the other
                          if (text.includes(accumulatedText)) {
                            // Received text contains accumulated - use received (they're same length so this shouldn't happen)
                            console.warn(`[Gemini Stream] Received contains accumulated, but same length - keeping accumulated`);
                          } else if (accumulatedText.includes(text)) {
                            // Accumulated contains received - keep accumulated
                            console.warn(`[Gemini Stream] Accumulated contains received - keeping accumulated`);
                          }
                        }
                      }
                    }
                  }
                }
              } catch (parseError) {
                // Skip malformed JSON lines
                if (parseError instanceof SyntaxError) {
                  continue;
                }
                // Re-throw API errors
                throw parseError;
              }
            }
          }
          
          // Only exit loop when reader says done
          if (done) {
            console.log(`[Gemini Stream] Reader done. Processed all chunks. Final text length: ${accumulatedText.length}`);
            break;
          }
        }
        
        // Process any remaining buffer after stream ends
        // This ensures we don't miss any final chunks
        if (buffer.trim()) {
          const trimmed = buffer.trim();
          if (trimmed.startsWith('data: ')) {
            try {
              const jsonStr = trimmed.slice(6);
              const data = JSON.parse(jsonStr);
              
              if (data.error) {
                throw new Error(`Gemini API error: ${data.error.message || JSON.stringify(data.error)}`);
              }
              
              const candidates = data?.candidates;
              if (candidates && candidates.length > 0) {
                const candidate = candidates[0];
                
                // Process content parts first
                if (candidate.content?.parts && candidate.content.parts.length > 0) {
                  for (const part of candidate.content.parts) {
                    const text = part.text || "";
                    if (text) {
                      // Use same simplified logic: always use longest text
                      if (text.length > accumulatedText.length) {
                        const delta = text.slice(accumulatedText.length);
                        hasYieldedContent = true;
                        console.log(`[Gemini Stream] Final buffer: received ${delta.length} new characters (${accumulatedText.length} → ${text.length})`);
                        // Yield two characters at a time with 25ms delay between chunks
                        for (let i = 0; i < delta.length; i += 2) {
                          const chunk = delta.slice(i, i + 2);
                          yield { content: chunk, done: false };
                          // 25ms delay between chunks for readable streaming
                          if (i < delta.length - 2) {
                            await new Promise(resolve => setTimeout(resolve, 25));
                          }
                        }
                        accumulatedText = text;
                      } else if (accumulatedText.length === 0) {
                        // First chunk in final buffer - send immediately
                        hasYieldedContent = true;
                        // Yield two characters at a time with 25ms delay between chunks
                        for (let i = 0; i < text.length; i += 2) {
                          const chunk = text.slice(i, i + 2);
                          yield { content: chunk, done: false };
                          // 25ms delay between chunks for readable streaming
                          if (i < text.length - 2) {
                            await new Promise(resolve => setTimeout(resolve, 25));
                          }
                        }
                        accumulatedText = text;
                      } else if (text.length < accumulatedText.length) {
                        // Shorter text - might be a different segment
                        if (!accumulatedText.includes(text) && !text.includes(accumulatedText)) {
                          // Different segment - concatenate it with proper spacing
                          console.warn(`[Gemini Stream] Final buffer: Received different segment (${text.length} chars). Concatenating.`);

                          // Check if we need a space between segments
                          const needsSpace = accumulatedText.length > 0 &&
                            !accumulatedText.endsWith(' ') &&
                            !accumulatedText.endsWith('\n') &&
                            !text.startsWith(' ') &&
                            !text.startsWith('\n');

                          if (needsSpace) {
                            yield { content: ' ', done: false };
                            accumulatedText += ' ';
                          }

                          hasYieldedContent = true;
                          for (let i = 0; i < text.length; i += 2) {
                            const chunk = text.slice(i, i + 2);
                            yield { content: chunk, done: false };
                            // 25ms delay between chunks for readable streaming
                            if (i < text.length - 2) {
                              await new Promise(resolve => setTimeout(resolve, 25));
                            }
                          }
                          accumulatedText += text;
                        } else {
                          console.log(`[Gemini Stream] Final buffer: text length ${text.length} <= accumulated ${accumulatedText.length}, skipping (subset)`);
                        }
                      } else {
                        console.log(`[Gemini Stream] Final buffer: text length ${text.length} <= accumulated ${accumulatedText.length}, skipping`);
                      }
                    }
                  }
                }
              }
            } catch (e) {
              if (e instanceof Error && e.message.includes("Gemini API")) {
                throw e;
              }
              // Log parse errors but don't fail
              console.warn(`[Gemini Stream] Error processing final buffer:`, e);
            }
          }
        }
        
        // If we got no content at all, try next model
        if (!hasYieldedContent) {
          console.error(`[Gemini] No content received from ${model}. Response was empty.`);
          if (model === models[models.length - 1]) {
            throw new Error(`Gemini API returned no data. Model: ${model}. Check API key and model availability.`);
          }
          continue;
        }
        
        // Success - log final state and yield done signal
        // Now that we've processed ALL chunks, we can safely mark as done
        console.log(`[Gemini Stream] Stream complete. Finish reason seen: ${seenFinishReason}, Final text length: ${accumulatedText.length}`);
        console.log(`[Gemini Stream] Final text preview: "${accumulatedText.substring(0, 200)}${accumulatedText.length > 200 ? '...' : ''}"`);
        console.log(`[Gemini Stream] Full text length: ${accumulatedText.length} characters`);
        yield { content: "", done: true };
        return;
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.error(`[Gemini] Error with model ${model}:`, lastError.message);
      // Continue to next model if available
      if (model === models[models.length - 1]) {
        throw lastError;
      }
    }
  }

  // Should not reach here, but just in case
  throw lastError || new Error("Failed to connect to Gemini API");
}

/**
 * Stream response from OpenAI API
 */
export async function* streamOpenAIResponse(
  messages: ChatMessage[],
  systemPrompt: string,
  apiKey: string
): AsyncGenerator<StreamChunk, void, unknown> {
  const openAIMessages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    ...messages.filter(m => m.role !== "system"),
  ];

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: openAIMessages,
      stream: true,
      temperature: 0.7,
      max_tokens: 1024,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error: ${response.status} ${error}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body reader");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") {
            yield { content: "", done: true };
            return;
          }

          try {
            const json = JSON.parse(data);
            const content = json.choices?.[0]?.delta?.content || "";
            if (content) {
              yield { content, done: false };
            }
          } catch (e) {
            // Skip malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield { content: "", done: true };
}

/**
 * Get the appropriate LLM provider based on available API keys
 * Works in both Node.js and Edge runtime
 */
export function getProvider(apiKey?: string): LLMProvider {
  // If API key passed directly (from route), use it
  if (apiKey) {
    // Determine provider from context or default to gemini
    return "gemini";
  }

  // Otherwise, check environment variables
  // Support both GEMINI_API_KEY and GOOGLE_AI_API_KEY for compatibility
  const geminiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  // Prefer Gemini, fallback to OpenAI
  if (geminiKey) return "gemini";
  if (openaiKey) return "openai";
  
  throw new Error("No LLM API key found. Set GEMINI_API_KEY or OPENAI_API_KEY");
}

/**
 * Stream response from the configured provider
 */
export async function* streamLLMResponse(
  messages: ChatMessage[],
  systemPrompt: string,
  providedApiKey?: string
): AsyncGenerator<StreamChunk, void, unknown> {
  const provider = getProvider(providedApiKey);
  
  // Get API key - prefer provided, then env vars
  // Support both GEMINI_API_KEY and GOOGLE_AI_API_KEY for compatibility
  const geminiKey = providedApiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY;
  const openaiKey = providedApiKey || process.env.OPENAI_API_KEY;
  
  const apiKey = provider === "gemini" ? geminiKey : openaiKey;

  if (!apiKey) {
    throw new Error(`API key not found for provider: ${provider}. Please set GEMINI_API_KEY in .env.local`);
  }

  if (provider === "gemini") {
    yield* streamGeminiResponse(messages, systemPrompt, apiKey);
  } else {
    yield* streamOpenAIResponse(messages, systemPrompt, apiKey);
  }
}
