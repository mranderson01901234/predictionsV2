/**
 * Test script to verify Gemini API connection
 * Run with: node test-gemini-api.js
 */

const API_KEY = "AIzaSyAY5be8-y7SgQHlaxjoPqOzX2lULdXxqcY";

async function testGeminiAPI() {
  const models = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-pro"];
  
  for (const model of models) {
    console.log(`\nTesting model: ${model}`);
    
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${API_KEY}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            contents: [{
              role: "user",
              parts: [{ text: "Say hello in one word" }],
            }],
            generationConfig: {
              maxOutputTokens: 10,
            },
          }),
        }
      );

      console.log(`Status: ${response.status}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Error: ${errorText}`);
        continue;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let hasContent = false;
      let rawChunks = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        rawChunks.push(buffer);
        
        // Try different parsing strategies
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
          console.log(`Raw chunk: ${chunk.substring(0, 100)}...`);
          
          if (chunk.startsWith("data: ")) {
            const jsonStr = chunk.slice(6).trim();
            if (jsonStr === "[DONE]") {
              console.log("✓ Stream complete");
              break;
            }
            try {
              const data = JSON.parse(jsonStr);
              console.log("Parsed data:", JSON.stringify(data, null, 2).substring(0, 200));
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
              if (text) {
                process.stdout.write(text);
                hasContent = true;
              }
            } catch (e) {
              console.log("JSON parse error:", e.message);
            }
          } else if (chunk.trim()) {
            // Try parsing as direct JSON (not SSE format)
            try {
              const data = JSON.parse(chunk);
              console.log("Direct JSON:", JSON.stringify(data, null, 2).substring(0, 200));
              const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
              if (text) {
                process.stdout.write(text);
                hasContent = true;
              }
            } catch (e) {
              // Not JSON, skip
            }
          }
        }
      }
      
      if (!hasContent && rawChunks.length > 0) {
        console.log("\nFirst raw chunk:", rawChunks[0].substring(0, 500));
      }

      if (hasContent) {
        console.log(`\n✓ Model ${model} works!`);
        return model;
      } else {
        console.log(`✗ Model ${model} returned no content`);
      }
    } catch (error) {
      console.log(`✗ Error: ${error.message}`);
    }
  }
  
  console.log("\n✗ No working model found");
  return null;
}

testGeminiAPI().then((workingModel) => {
  if (workingModel) {
    console.log(`\n✅ Use model: ${workingModel}`);
  } else {
    console.log("\n❌ All models failed. Check your API key.");
  }
});

