import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";

const BLIND_ASSISTANT_SYSTEM_PROMPT = `You are a conversational AI navigation guide assisting a completely blind person navigating inside a building.
The user is holding a camera on their device providing your live view or short environment video clip, and speaking to you naturally.

CRITICAL INSTRUCTIONS FOR ASSISTING A BLIND PERSON:
1. ENVIRONMENT INSPECTION: On every user query, thoroughly examine the camera snapshot or video clip for notice boards, door signs, room numbers, exit signs, open corridors, doors, walls, and obstacles.
2. IF TARGET OR CLEAR PATH IS VISIBLE:
   - Provide direct, concise walking guidance (e.g., 'Straight ahead is the door to Room 204. Walk forward 5 steps.', 'The corridor branches to your right. Take 3 steps forward and turn right.').
   - Set "needsMoreInfo": false.
3. IF INFORMATION IS MISSING OR CAMERA VIEW IS OBSTRUCTED (CRUCIAL):
   - Because the user is blind, they cannot see where their camera is aiming.
   - If the view shows a blank wall, floor, corner, or lacks signs to answer their question:
     PROACTIVELY INSTRUCT THE USER TO TURN OR ADJUST THEIR POSITION.
   - Set "needsMoreInfo": true.
   - Examples of explicit physical instructions:
     * 'I see a blank wall ahead. Please turn 90 degrees to your right so I can inspect the corridor.'
     * 'The camera is tilted toward the ground. Please raise your camera straight ahead.'
     * 'No signs or doors found in this direction. Please turn around 180 degrees so I can look behind you.'
4. SPEECH OPTIMIZED & JSON FORMAT:
   - The user cannot see the screen; your response will be read ALOUD via Text-to-Speech.
   - Do NOT use markdown asterisks (*), hashtags (#), or bullet points.
   - Keep guidance concise, natural, and under 2 short spoken sentences.
   - Return your output ONLY as a valid JSON object matching this schema:
   {
     "guidance": "spoken response text for the blind user",
     "needsMoreInfo": true or false,
     "actionPrompt": "turn_right" | "turn_left" | "turn_around" | "raise_camera" | "none"
   }`;

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        {
          success: false,
          error: "GEMINI_API_KEY is missing from environment variables.",
        },
        { status: 500 }
      );
    }

    const body = await req.json();
    const { image, mediaData, mimeType, prompt, chatHistory, detectedObstacles } = body;

    const rawData = mediaData || image;
    if (!rawData) {
      return NextResponse.json(
        { success: false, error: "No camera snapshot or environment video clip provided." },
        { status: 400 }
      );
    }

    const cleanBase64 = rawData.replace(/^data:[^;]+;base64,/, "");
    const effectiveMime = mimeType || "image/jpeg";

    const genAI = new GoogleGenerativeAI(apiKey);
    const candidateModels = [
      "gemini-3.5-flash",
      "gemini-3.7-flash",
      "gemini-3.8-flash",
      "gemini-2.5-flash",
      "gemini-flash-latest",
    ];

    // Build conversational context from recent chat history
    let historyContext = "";
    if (chatHistory && Array.isArray(chatHistory) && chatHistory.length > 0) {
      historyContext = "Recent conversation:\n" +
        chatHistory
          .slice(-4)
          .map((m: any) => `${m.sender === "user" ? "Blind User" : "AI Guide"}: ${m.text}`)
          .join("\n") + "\n\n";
    }

    let obstacleContext = "";
    if (detectedObstacles && Array.isArray(detectedObstacles) && detectedObstacles.length > 0) {
      const obstacleSummary = detectedObstacles
        .map((o: any) => `${o.class} (${o.direction}, ${o.proximity.toLowerCase()})`)
        .join("; ");
      obstacleContext = `\n(Local proximity sensors detect: ${obstacleSummary})`;
    }

    const currentPrompt = prompt && prompt.trim().length > 0
      ? `Blind user asks: "${prompt}".\nInspect the environment video/image. If you see what they need, guide them directly. If the view is obstructed, facing a wall, or lacks signs, instruct them to turn or adjust and set needsMoreInfo to true.`
      : "Please inspect this environment view. Tell me what is around me, read any visible boards or door signs, and if needed, tell me which direction to turn.";

    const fullPrompt = `${historyContext}${currentPrompt}${obstacleContext}\nRespond with JSON: {"guidance": "...", "needsMoreInfo": boolean, "actionPrompt": "..."}`;

    const mediaPart = {
      inlineData: {
        data: cleanBase64,
        mimeType: effectiveMime,
      },
    };

    let result;
    let successfulModel = "";
    let lastError: any = null;

    for (const modelName of candidateModels) {
      try {
        const model = genAI.getGenerativeModel({
          model: modelName,
          systemInstruction: BLIND_ASSISTANT_SYSTEM_PROMPT,
        });
        result = await model.generateContent([mediaPart, fullPrompt]);
        successfulModel = modelName;
        break;
      } catch (err: any) {
        lastError = err;
        console.warn(`Model ${modelName} failed:`, err.message);
      }
    }

    if (!result) {
      throw lastError || new Error("Failed to generate response with available Gemini models.");
    }

    const rawResponseText = result.response.text().trim();

    // Parse structured JSON or fallback to clean string
    let parsedGuidance = "";
    let needsMoreInfo = false;
    let actionPrompt = "none";

    try {
      const jsonMatch = rawResponseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        parsedGuidance = parsed.guidance || "";
        needsMoreInfo = Boolean(parsed.needsMoreInfo);
        actionPrompt = parsed.actionPrompt || "none";
      }
    } catch {}

    if (!parsedGuidance) {
      parsedGuidance = rawResponseText
        .replace(/[*#_`~]/g, "")
        .replace(/\{[\s\S]*\}/g, "")
        .replace(/\n+/g, " ")
        .trim();

      // Heuristic detection if JSON parsing wasn't clean
      const lower = parsedGuidance.toLowerCase();
      if (lower.includes("turn ") || lower.includes("raise ") || lower.includes("tilt ") || lower.includes("look around")) {
        needsMoreInfo = true;
      }
    }

    const cleanSpeech = parsedGuidance.replace(/[*#_`~]/g, "").trim();

    return NextResponse.json({
      success: true,
      guidance: cleanSpeech,
      needsMoreInfo,
      actionPrompt,
      modelUsed: successfulModel,
    });
  } catch (err: unknown) {
    const error = err as Error;
    console.error("Gemini Vision Guide error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || "Failed to process media with Gemini Vision.",
      },
      { status: 500 }
    );
  }
}
