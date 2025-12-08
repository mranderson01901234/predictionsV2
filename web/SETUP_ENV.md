# Environment Setup for AI Chat

## Quick Setup

1. **Create `.env.local` file** in the `web/` directory:

```bash
cd web
touch .env.local
```

2. **Add your API key** to `.env.local`:

```bash
GEMINI_API_KEY=AIzaSyDSATpzoJ1iREop-GoRHnHS-Lmn3zyKpXA
```

Or manually edit the file and add:
```
GEMINI_API_KEY=AIzaSyDSATpzoJ1iREop-GoRHnHS-Lmn3zyKpXA
```

**Note:** The code also supports `GOOGLE_AI_API_KEY` for backward compatibility, but `GEMINI_API_KEY` is the recommended name per Google's documentation.

3. **Restart your dev server**:

```bash
# Stop the current server (Ctrl+C)
# Then restart:
npm run dev
```

## Model Configuration

The chat uses **gemini-2.5-flash** as the primary model, with automatic fallback to `gemini-2.0-flash-exp` if needed.

## Verify Setup

After restarting, check the browser console. You should see:
- No 404 errors for the API route
- Chat messages should get responses

## Troubleshooting

### If chat still doesn't work:

1. **Check the API key is loaded**:
   - Open browser DevTools → Network tab
   - Send a chat message
   - Check the `/api/chat` request
   - Look for error messages in the response

2. **Check server logs**:
   - Look at the terminal where `npm run dev` is running
   - Check for any error messages

3. **Test the API directly**:
   ```bash
   curl -X POST http://localhost:3000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
   ```

4. **Verify environment variable**:
   ```bash
   # In the web directory
   cat .env.local
   # Should show: GEMINI_API_KEY=your_key_here
   ```

## Security Note

The `.env.local` file is already in `.gitignore` and will NOT be committed to git. Your API key is safe.
