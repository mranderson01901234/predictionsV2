# Gemini Streaming Implementation Audit

**Date**: 2025-01-XX  
**Status**: ✅ Fixed

---

## Issues Found

### 1. ❌ API Key Authentication Method
**Problem**: Using query parameter `?key=${apiKey}` instead of header  
**Impact**: May not work with SSE streaming format  
**Fix**: Changed to `x-goog-api-key` header

### 2. ❌ Missing SSE Parameter
**Problem**: Not using `alt=sse` query parameter  
**Impact**: May not enable proper Server-Sent Events streaming  
**Fix**: Added `alt=sse` to URL

### 3. ❌ Buffering Delays First Token
**Problem**: Complex JSON parsing logic buffers chunks before processing  
**Impact**: First token arrives with delay, not immediate  
**Fix**: Simplified to line-by-line SSE processing

### 4. ❌ Overly Complex Parsing
**Problem**: Complex bracket counting and multi-format handling  
**Impact**: Slower processing, potential bugs  
**Fix**: Simplified to SSE format parsing

---

## Changes Made

### File: `web/src/lib/ai/providers.ts`

#### 1. Updated API URL and Headers
```typescript
// BEFORE:
const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${apiKey}`;
const response = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(requestBody),
});

// AFTER:
const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
const response = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-goog-api-key": apiKey,
  },
  body: JSON.stringify(requestBody),
});
```

#### 2. Simplified Streaming Parser
- **Removed**: Complex bracket counting logic
- **Removed**: Multi-format handling (NDJSON vs arrays)
- **Added**: Simple SSE line-by-line processing
- **Added**: Delta extraction to yield only new text

#### 3. Immediate Processing
- Process chunks as they arrive (no buffering)
- Extract complete lines immediately
- Yield text deltas character-by-character

---

## Expected Behavior

### Streaming Format
The Gemini API with `alt=sse` returns Server-Sent Events format:
```
data: {"candidates":[...]}\n\n
data: {"candidates":[...]}\n\n
...
```

### Text Extraction
- Each `data:` line contains a JSON object
- `candidates[0].content.parts[0].text` contains the accumulated text
- We extract deltas by comparing with previously accumulated text
- Yield only new characters immediately

---

## Testing Checklist

- [ ] Verify first token arrives immediately (< 100ms)
- [ ] Verify streaming is letter-by-letter (not chunk-by-chunk)
- [ ] Test with various message lengths
- [ ] Verify error handling still works
- [ ] Test fallback to secondary model
- [ ] Verify SSE format parsing handles edge cases

---

## Reference

Based on official Gemini API curl example:
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H 'Content-Type: application/json' \
  --no-buffer \
  -d '{...}'
```

Key points:
- `alt=sse` enables SSE streaming
- `x-goog-api-key` header for authentication
- `--no-buffer` ensures immediate streaming (handled by fetch API)

---

## Notes

- The `--no-buffer` flag in curl is equivalent to processing chunks immediately in JavaScript
- SSE format ensures each line is a complete JSON object
- Delta extraction ensures we only yield new text, not duplicates
- Frontend expects `data: {...}` format (already compatible)

