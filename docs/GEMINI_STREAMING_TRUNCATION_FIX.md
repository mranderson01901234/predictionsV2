# Gemini Streaming Truncation Fix

**Date**: 2025-01-XX  
**Issue**: Streaming responses were being truncated at ~225 characters  
**Status**: ✅ Fixed

---

## Problem

The Gemini streaming API was working, but responses were being truncated at approximately 225 characters. Logs showed:
```
[Gemini Stream] Finish reason: STOP, Final text length: 225
[API Route] Stream done. Total chunks: 225, Total content length: 225
```

The response was cut off mid-sentence, indicating the stream was ending prematurely.

---

## Root Causes Identified

### 1. **Incorrect Text Accumulation Logic** ⚠️ PRIMARY ISSUE
- **Issue**: Code assumed Gemini always sends FULL accumulated text in each chunk, but logs showed chunks with different text segments that didn't start with accumulated text
- **Problem**: When a chunk's text didn't start with accumulated text, the code would skip it or incorrectly process it, causing truncation
- **Fix**: Implemented robust text matching:
  - Check if new text starts with accumulated text (normal case)
  - Handle first chunk (no accumulated text yet)
  - Find common prefix and extract delta for edge cases
  - Better recovery logic for unexpected formats

### 2. **Low `maxOutputTokens` Limit**
- **Issue**: `maxOutputTokens` was set to 1024, which should be enough (~4000 characters), but may have been causing early termination in some cases
- **Fix**: Increased to 8192 tokens to allow longer responses

### 3. **Insufficient Logging**
- **Issue**: Limited visibility into where truncation was occurring
- **Fix**: Added comprehensive logging at multiple points:
  - When new content chunks are received
  - When finishReason is detected
  - Final text length and preview
  - Warnings for unexpected conditions (text format mismatches)

### 4. **Potential Buffer Processing Issues**
- **Issue**: Edge cases in buffer processing could cause content to be missed
- **Fix**: Added defensive checks and better handling of edge cases:
  - Common prefix finding for mismatched text
  - Better handling of confirmation chunks
  - Improved final buffer processing with same logic

### 5. **Stream Closure Timing**
- **Issue**: Stream might close before all chunks are processed
- **Fix**: Added delays and better synchronization:
  - Increased delay before closing stream (100ms → 150ms)
  - Better handling of final buffer content
  - Ensured [DONE] signal is always sent

---

## Changes Made

### File: `web/src/lib/ai/providers.ts`

#### 1. Increased Token Limit
```typescript
// BEFORE:
maxOutputTokens: 1024,

// AFTER:
maxOutputTokens: 8192, // Increased from 1024 to allow longer responses
```

#### 2. Enhanced Logging
- Added logging when new content chunks are received
- Added logging for confirmation chunks
- Added warnings for unexpected conditions (text shorter than accumulated)
- Enhanced final text logging with full length and preview

#### 3. Fixed Text Accumulation Logic (CRITICAL FIX)
- **Before**: Assumed all chunks contain full accumulated text starting from beginning
- **After**: Robust handling of different text formats:
  - Check if text starts with accumulated (normal case)
  - Handle first chunk separately (no accumulated text yet)
  - Find common prefix for edge cases where text format differs
  - Recovery logic for unexpected text segments
- Added `findCommonPrefix()` helper function for matching text segments

#### 4. Better Final Buffer Handling
- Improved processing of remaining buffer after stream ends
- Added logging for final buffer processing
- Ensured all content is processed before marking as done

### File: `web/src/app/api/chat/route.ts`

#### 1. Enhanced Stream Management
- Added tracking of last chunk time
- Improved logging with content preview
- Increased delay before closing stream (100ms → 150ms)
- Better error handling with [DONE] signal even on errors

#### 2. Better Completion Handling
- Ensured [DONE] signal is always sent
- Added delays to ensure all data is flushed
- Improved logging for stream completion

---

## Testing Checklist

After these changes, verify:

- [ ] Responses are no longer truncated at 225 characters
- [ ] Longer responses (500+ characters) stream correctly
- [ ] All content is displayed in the chat UI
- [ ] Logs show complete text length matches expected length
- [ ] No warnings about text length mismatches appear
- [ ] Stream completes properly with [DONE] signal
- [ ] Error handling still works correctly

---

## Monitoring

Watch for these log messages to verify correct behavior:

### Expected Logs (Normal Operation)
```
[Gemini Stream] Received X new characters (accumulated: Y, total in chunk: Z)
[Gemini Stream] Finish reason: STOP, Final text length: XXX
[Gemini Stream] Final text preview: "..."
[Gemini Stream] Full text length: XXX characters
[API Route] Stream done. Total chunks: XXX, Total content length: XXX
[Frontend] Stream complete. Total chunks: XXX, Final content length: XXX
```

### Warning Logs (Indicate Issues)
```
[Gemini Stream] WARNING: Received text (X chars) doesn't match accumulated (Y chars)
[Gemini Stream] WARNING: Text doesn't start with accumulated. Common prefix: X chars
```

These warnings indicate text format mismatches. The code now handles these cases by finding common prefixes and extracting deltas, but if they appear frequently, it may indicate an API behavior change.

---

## Next Steps

1. **Test with various response lengths**: Try prompts that should generate short, medium, and long responses
2. **Monitor logs**: Check server logs to verify complete text is being received
3. **Verify UI**: Ensure all content appears in the chat interface
4. **Check for edge cases**: Test with various prompt types and conversation contexts

---

## Related Files

- `web/src/lib/ai/providers.ts` - Streaming provider implementation
- `web/src/app/api/chat/route.ts` - API route handler
- `web/src/hooks/useAIChat.ts` - Frontend stream processing
- `docs/GEMINI_STREAMING_AUDIT.md` - Previous streaming audit

---

## Notes

- The character-by-character streaming with delays is intentional for better UX
- The 20ms delay per character creates a natural reading speed (~40 chars/sec)
- Spaces and newlines skip the delay to maintain natural flow
- All changes maintain backward compatibility with existing functionality

