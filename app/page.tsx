"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";

interface NavigationMessage {
  id: string;
  sender: "user" | "gemini" | "coco";
  text: string;
  timestamp: string;
  isAlert?: boolean;
}

export interface TrackedObject {
  id: string;
  class: string;
  score: number;
  bbox: [number, number, number, number];
  area: number;
  areaDelta: number;
  proximity: "APPROACHING" | "RECEDING" | "STABLE";
  direction: "LEFT" | "CENTER" | "RIGHT";
  screenX: number;
  screenY: number;
}

type ConversationalState = "idle" | "listening" | "user_speaking" | "thinking" | "ai_speaking" | "reinspecting";

const WALKING_HAZARDS = new Set([
  "person",
  "chair",
  "couch",
  "sofa",
  "bed",
  "dining table",
  "table",
  "bench",
  "bicycle",
  "motorcycle",
  "car",
  "stairs",
  "suitcase",
  "door",
]);

export default function BlindNavigationPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectionCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Audio & Speech Refs
  const recognitionRef = useRef<any>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const speechAccumulatorRef = useRef<string>("");
  const chatHistoryRef = useRef<Array<{ sender: "user" | "gemini"; text: string }>>([]);
  const isAiSpeakingRef = useRef<boolean>(false);
  const micActiveRef = useRef<boolean>(true);
  const isUserSpeakingRef = useRef<boolean>(false);
  const userQueryActiveRef = useRef<boolean>(false);
  const cooldownTimerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isAcquiringMediaRef = useRef<boolean>(false);

  // Decoupled Hardware & Audio Pipeline Refs
  const audioStreamRef = useRef<MediaStream | null>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const volumeAnimFrameRef = useRef<number | null>(null);
  const isRecognitionActiveRef = useRef<boolean>(false);
  const recognitionRestartTimerRef = useRef<NodeJS.Timeout | null>(null);
  const cameraRetryTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Proactive Re-Inspection & Video Analysis Refs
  const reinspectTimerRef = useRef<NodeJS.Timeout | null>(null);
  const reinspectIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reinspectAttemptsRef = useRef<number>(0);
  const clearReinspectTimersRef = useRef<(() => void) | null>(null);
  const dispatchConversationalQueryRef = useRef<any>(null);

  // TensorFlow / COCO-SSD Refs
  const modelRef = useRef<any>(null);
  const detectionLoopRef = useRef<number | null>(null);
  const isDetectingRef = useRef<boolean>(false);
  const trackedHistoryRef = useRef<Record<string, { smoothedArea: number; lastSeen: number }>>({});
  const lastSpokenAlertRef = useRef<number>(0);
  const currentObjectsRef = useRef<TrackedObject[]>([]);

  // States
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [micActive, setMicActive] = useState<boolean>(true);
  const [isMirrored, setIsMirrored] = useState<boolean>(false);
  const [voiceGuidanceEnabled, setVoiceGuidanceEnabled] = useState<boolean>(true);

  const [cocoModelReady, setCocoModelReady] = useState<boolean>(false);
  const [trackedObjects, setTrackedObjects] = useState<TrackedObject[]>([]);

  const [conversationalState, setConversationalState] = useState<ConversationalState>("listening");
  const conversationalStateRef = useRef<ConversationalState>("listening");
  const [currentLiveSpeech, setCurrentLiveSpeech] = useState<string>("");
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const [navigationLogs, setNavigationLogs] = useState<NavigationMessage[]>([]);
  const [reinspectCountdown, setReinspectCountdown] = useState<number | null>(null);

  const [cameraNotice, setCameraNotice] = useState<string | null>(null);
  const [speechNotice, setSpeechNotice] = useState<string | null>(null);
  const [manualQueryText, setManualQueryText] = useState<string>("");

  useEffect(() => {
    conversationalStateRef.current = conversationalState;
  }, [conversationalState]);

  // Clean Reinspection Timers
  const clearReinspectTimers = useCallback(() => {
    if (reinspectIntervalRef.current) {
      clearInterval(reinspectIntervalRef.current);
      reinspectIntervalRef.current = null;
    }
    if (reinspectTimerRef.current) {
      clearTimeout(reinspectTimerRef.current);
      reinspectTimerRef.current = null;
    }
    setReinspectCountdown(null);
  }, []);

  clearReinspectTimersRef.current = clearReinspectTimers;

  // 1. Text-to-Speech Voice Guidance (with onCompleted callback support)
  const speakGuidance = useCallback(
    (text: string, isEmergency = false, onCompleted?: () => void) => {
      if (!voiceGuidanceEnabled || typeof window === "undefined" || !("speechSynthesis" in window)) {
        if (onCompleted) {
          onCompleted();
        }
        return;
      }
      try {
        if (isEmergency) {
          window.speechSynthesis.cancel();
          isAiSpeakingRef.current = false;
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = isEmergency ? 0.9 : 1.0;
        utterance.pitch = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const clearVoice =
          voices.find((v) => v.lang.startsWith("en") && (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Samantha"))) ||
          voices.find((v) => v.lang.startsWith("en"));
        if (clearVoice) {
          utterance.voice = clearVoice;
        }

        utterance.onstart = () => {
          if (!isEmergency) {
            isAiSpeakingRef.current = true;
            setConversationalState("ai_speaking");
          }
        };

        utterance.onend = () => {
          if (!isEmergency) {
            isAiSpeakingRef.current = false;
            isUserSpeakingRef.current = false;
            setConversationalState("listening");
            speechAccumulatorRef.current = "";
            setCurrentLiveSpeech("");
            if (micActiveRef.current) {
              try {
                recognitionRef.current?.start();
              } catch {}
            }

            if (onCompleted) {
              onCompleted();
            } else {
              // 3-second quiet absorption cooldown before normal guidance/alerts resume
              if (cooldownTimerRef.current) clearTimeout(cooldownTimerRef.current);
              cooldownTimerRef.current = setTimeout(() => {
                userQueryActiveRef.current = false;
              }, 3000);
            }
          }
        };

        utterance.onerror = () => {
          isAiSpeakingRef.current = false;
          isUserSpeakingRef.current = false;
          setConversationalState("listening");
          if (micActiveRef.current) {
            try {
              recognitionRef.current?.start();
            } catch {}
          }
          if (onCompleted) {
            onCompleted();
          } else {
            if (cooldownTimerRef.current) clearTimeout(cooldownTimerRef.current);
            cooldownTimerRef.current = setTimeout(() => {
              userQueryActiveRef.current = false;
            }, 3000);
          }
        };

        window.speechSynthesis.speak(utterance);
      } catch (err) {
        console.error("Speech error:", err);
        isAiSpeakingRef.current = false;
        setConversationalState("listening");
        if (onCompleted) onCompleted();
      }
    },
    [voiceGuidanceEnabled]
  );

  // 2. High-Resolution Frame Screenshot
  const captureFrameSnapshot = (): string | null => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = video.videoWidth || 640;
    offscreenCanvas.height = video.videoHeight || 480;
    const ctx = offscreenCanvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    return offscreenCanvas.toDataURL("image/jpeg", 0.85);
  };

  // 3. Send Captured Screenshot & Speech Query to Gemini Vision
  const dispatchConversationalQuery = useCallback(
    async (userPromptText: string, isAutoFollowUp = false) => {
      if (!userPromptText || userPromptText.trim().length === 0) {
        isUserSpeakingRef.current = false;
        userQueryActiveRef.current = false;
        setConversationalState("listening");
        return;
      }

      clearReinspectTimers();
      const cleanQuestion = userPromptText.trim();
      userQueryActiveRef.current = true;
      speechAccumulatorRef.current = "";
      setCurrentLiveSpeech("");
      setConversationalState("thinking");

      if (!isAutoFollowUp) {
        reinspectAttemptsRef.current = 0;
        const userMsg: NavigationMessage = {
          id: String(Date.now()),
          sender: "user",
          text: cleanQuestion,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };
        setNavigationLogs((prev) => [...prev, userMsg]);
        chatHistoryRef.current.push({ sender: "user", text: cleanQuestion });
      } else {
        reinspectAttemptsRef.current += 1;
        chatHistoryRef.current.push({ sender: "user", text: cleanQuestion });
      }

      // Instantly capture camera screenshot (no video delay)
      let snapshot = captureFrameSnapshot();
      if (!snapshot) {
        // Fallback 1x1 black frame so Gemini still processes user navigation question
        snapshot =
          "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=";
      }

      try {
        const res = await fetch("/api/vision-guide", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: snapshot,
            mimeType: "image/jpeg",
            prompt: cleanQuestion,
            chatHistory: chatHistoryRef.current,
            detectedObstacles: currentObjectsRef.current,
          }),
        });

        const data = await res.json();

        if (data.success && data.guidance) {
          const responseText = data.guidance;
          chatHistoryRef.current.push({ sender: "gemini", text: responseText });
          if (chatHistoryRef.current.length > 8) {
            chatHistoryRef.current = chatHistoryRef.current.slice(-8);
          }

          setNavigationLogs((prev) => [
            ...prev,
            {
              id: String(Date.now()),
              sender: "gemini",
              text: responseText,
              timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            },
          ]);

          const shouldReinspect = Boolean(data.needsMoreInfo) && reinspectAttemptsRef.current < 2;

          if (shouldReinspect) {
            // Speak physical turn/orientation instruction, then start auto re-inspection countdown
            speakGuidance(responseText, false, () => {
              clearReinspectTimers();
              setConversationalState("reinspecting");
              userQueryActiveRef.current = true;
              let seconds = 3;
              setReinspectCountdown(seconds);

              reinspectIntervalRef.current = setInterval(() => {
                seconds -= 1;
                if (seconds > 0) {
                  setReinspectCountdown(seconds);
                } else {
                  clearReinspectTimers();
                  setReinspectCountdown(null);
                  dispatchConversationalQuery(
                    "I have turned and adjusted my position as requested. What do you see now?",
                    true
                  );
                }
              }, 1000);
            });
          } else {
            speakGuidance(responseText, false);
          }
        } else {
          const fallback = data.error || "Could not inspect the scene clearly. Please ask again.";
          speakGuidance(fallback, false);
        }
      } catch {
        speakGuidance("Connection issue while inspecting the surroundings. Please ask again.", false);
      }
    },
    [clearReinspectTimers, speakGuidance]
  );

  dispatchConversationalQueryRef.current = dispatchConversationalQuery;

  // 4. Safe Asynchronous Speech Recognition Engine (With Watchdog & Auto-Reconnect)
  const safeStartRecognitionRef = useRef<(() => void) | null>(null);

  const safeStartRecognition = useCallback(() => {
    if (typeof window === "undefined" || !micActiveRef.current || isAiSpeakingRef.current) return;
    if (isRecognitionActiveRef.current) return;

    if (!recognitionRef.current) {
      startSpeechRecognition();
      return;
    }

    try {
      recognitionRef.current.start();
      isRecognitionActiveRef.current = true;
      setSpeechNotice(null);
    } catch (err: any) {
      if (err?.name === "InvalidStateError") {
        isRecognitionActiveRef.current = true;
      } else {
        console.warn("Speech recognition restart retry:", err?.message);
        startSpeechRecognition();
      }
    }
  }, []);

  safeStartRecognitionRef.current = safeStartRecognition;

  const startSpeechRecognition = useCallback(() => {
    if (typeof window === "undefined") return;

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setSpeechNotice("Speech recognition not supported in this browser. Use Chrome or Edge.");
      return;
    }

    try {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {}
      }

      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      const userLang = (typeof navigator !== "undefined" && navigator.language) || "en-US";
      recognition.lang = userLang;

      recognition.onstart = () => {
        isRecognitionActiveRef.current = true;
        setSpeechNotice(null);
      };

      recognition.onresult = (event: any) => {
        let interimText = "";
        let latestFinal = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const piece = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            latestFinal += piece;
          } else {
            interimText += piece;
          }
        }

        const combinedText = (speechAccumulatorRef.current + " " + latestFinal + " " + interimText).trim();

        if (combinedText.length > 0) {
          clearReinspectTimersRef.current?.();

          // Immediately pause/cancel any active warning or AI guidance utterance
          if (typeof window !== "undefined" && window.speechSynthesis) {
            if (window.speechSynthesis.speaking) {
              window.speechSynthesis.cancel();
            }
          }
          if (cooldownTimerRef.current) {
            clearTimeout(cooldownTimerRef.current);
          }
          isAiSpeakingRef.current = false;
          isUserSpeakingRef.current = true;
          userQueryActiveRef.current = true;

          setConversationalState("user_speaking");
          setCurrentLiveSpeech(combinedText);

          if (latestFinal.trim().length > 0) {
            speechAccumulatorRef.current = (speechAccumulatorRef.current + " " + latestFinal).trim();
          }

          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
          }

          silenceTimerRef.current = setTimeout(() => {
            const finalQueryToDispatch = speechAccumulatorRef.current || combinedText;
            dispatchConversationalQueryRef.current?.(finalQueryToDispatch, false);
          }, 1200);
        }
      };

      recognition.onerror = (e: any) => {
        if (e.error === "no-speech") return;
        if (e.error === "aborted") {
          isRecognitionActiveRef.current = false;
          return;
        }
        console.warn("Speech recognition notice:", e.error);
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
          setSpeechNotice("Mic permission needed • Click anywhere to grant");
          isRecognitionActiveRef.current = false;
        } else if (e.error === "network") {
          setSpeechNotice("Speech service network retry...");
        }
      };

      recognition.onend = () => {
        isRecognitionActiveRef.current = false;
        if (micActiveRef.current && !isAiSpeakingRef.current) {
          if (recognitionRestartTimerRef.current) clearTimeout(recognitionRestartTimerRef.current);
          recognitionRestartTimerRef.current = setTimeout(() => {
            safeStartRecognitionRef.current?.();
          }, 200);
        }
      };

      recognition.start();
      recognitionRef.current = recognition;
      isRecognitionActiveRef.current = true;
    } catch (err) {
      console.error("Speech recognition start error:", err);
      isRecognitionActiveRef.current = false;
    }
  }, [safeStartRecognition]);

  // 5. Load COCO-SSD Model
  useEffect(() => {
    let isCancelled = false;

    async function loadCocoSsdModel() {
      try {
        const tf = await import("@tensorflow/tfjs");
        await tf.ready();
        const cocoSsd = await import("@tensorflow-models/coco-ssd");
        const loadedModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });

        if (!isCancelled) {
          modelRef.current = loadedModel;
          setCocoModelReady(true);
        }
      } catch (err) {
        console.error("Model load error:", err);
      }
    }

    loadCocoSsdModel();

    return () => {
      isCancelled = true;
    };
  }, []);

  // 6. Minimal Clean Bounding Box Renderer
  const drawCleanBoundingBox = (
    ctx: CanvasRenderingContext2D,
    obj: TrackedObject,
    displayWidth: number,
    displayHeight: number,
    videoWidth: number,
    videoHeight: number,
    mirrored: boolean
  ) => {
    const scaleX = displayWidth / videoWidth;
    const scaleY = displayHeight / videoHeight;

    let [x, y, w, h] = obj.bbox;
    x = x * scaleX;
    y = y * scaleY;
    w = w * scaleX;
    h = h * scaleY;

    if (mirrored) {
      x = displayWidth - (x + w);
    }

    const isCenterApproaching = obj.proximity === "APPROACHING" && obj.direction === "CENTER";
    const strokeColor = isCenterApproaching ? "#ef4444" : "rgba(255, 255, 255, 0.75)";

    // Subtle clean box
    ctx.lineWidth = isCenterApproaching ? 3 : 1.5;
    ctx.strokeStyle = strokeColor;
    ctx.strokeRect(x, y, w, h);

    // Clean small pill
    const label = `${obj.class} ${obj.proximity === "APPROACHING" ? "• closer" : ""}`.trim();
    ctx.font = "500 11px system-ui, -apple-system, sans-serif";
    const textWidth = ctx.measureText(label).width + 10;

    ctx.fillStyle = isCenterApproaching ? "#ef4444" : "rgba(15, 17, 23, 0.85)";
    ctx.beginPath();
    ctx.roundRect(x, Math.max(0, y - 20), textWidth, 18, 4);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, Math.max(13, y - 7));
  };

  // 7. Continuous COCO-SSD Detection Loop (Unobtrusive)
  const runContinuousDetection = useCallback(() => {
    if (!videoRef.current || !modelRef.current || !detectionCanvasRef.current) {
      detectionLoopRef.current = requestAnimationFrame(runContinuousDetection);
      return;
    }

    const video = videoRef.current;
    const canvas = detectionCanvasRef.current;

    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      detectionLoopRef.current = requestAnimationFrame(runContinuousDetection);
      return;
    }

    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      detectionLoopRef.current = requestAnimationFrame(runContinuousDetection);
      return;
    }

    const now = performance.now();

    if (!isDetectingRef.current) {
      isDetectingRef.current = true;

      modelRef.current
        .detect(video)
        .then((predictions: any[]) => {
          isDetectingRef.current = false;
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const videoWidth = video.videoWidth;
          const videoHeight = video.videoHeight;
          const updatedTracked: TrackedObject[] = [];

          predictions.forEach((pred, idx) => {
            const [x, y, w, h] = pred.bbox;
            const currentArea = w * h;

            const centroidX = x + w / 2;
            const rawNormalizedX = centroidX / videoWidth;
            const screenX = isMirrored ? 1 - rawNormalizedX : rawNormalizedX;
            const screenY = (y + h / 2) / videoHeight;

            let direction: "LEFT" | "CENTER" | "RIGHT" = "CENTER";
            if (screenX < 0.38) direction = "LEFT";
            else if (screenX > 0.62) direction = "RIGHT";

            const trackKey = `${pred.class}_${Math.round(screenX * 3)}`;
            const prev = trackedHistoryRef.current[trackKey];

            let proximity: "APPROACHING" | "RECEDING" | "STABLE" = "STABLE";
            let areaDelta = 0;
            let smoothedArea = currentArea;

            if (prev) {
              smoothedArea = prev.smoothedArea * 0.7 + currentArea * 0.3;
              areaDelta = (currentArea - prev.smoothedArea) / (prev.smoothedArea || 1);
              if (areaDelta > 0.05) proximity = "APPROACHING";
              else if (areaDelta < -0.05) proximity = "RECEDING";
            }

            trackedHistoryRef.current[trackKey] = { smoothedArea, lastSeen: now };

            const trackedObj: TrackedObject = {
              id: `${pred.class}-${idx}`,
              class: pred.class,
              score: pred.score,
              bbox: [x, y, w, h],
              area: currentArea,
              areaDelta,
              proximity,
              direction,
              screenX,
              screenY,
            };

            updatedTracked.push(trackedObj);
            drawCleanBoundingBox(ctx, trackedObj, canvas.width, canvas.height, videoWidth, videoHeight, isMirrored);
          });

          Object.keys(trackedHistoryRef.current).forEach((key) => {
            if (now - trackedHistoryRef.current[key].lastSeen > 2000) {
              delete trackedHistoryRef.current[key];
            }
          });

          setTrackedObjects(updatedTracked);
          currentObjectsRef.current = updatedTracked;

          // Slow Emergency Obstacle Alert on Center Path (PAUSED when user speaks, during query & response, and 3s cooldown)
          const isVoiceBusy =
            userQueryActiveRef.current ||
            isUserSpeakingRef.current ||
            isAiSpeakingRef.current ||
            conversationalStateRef.current !== "listening";

          if (!isVoiceBusy && updatedTracked.length > 0) {
            // Strictly detect REAL walking hazards located directly in the center path
            const centerObstacle = updatedTracked.find(
              (o) =>
                o.direction === "CENTER" &&
                o.score >= 0.52 &&
                WALKING_HAZARDS.has(o.class.toLowerCase())
            );

            if (
              centerObstacle &&
              centerObstacle.proximity === "APPROACHING" &&
              now - lastSpokenAlertRef.current > 6000
            ) {
              lastSpokenAlertRef.current = now;
              const alertMsg = `Caution. ${centerObstacle.class} directly in front of you. Please pause, or step aside.`;
              speakGuidance(alertMsg, true);

              setNavigationLogs((prev) => [
                ...prev,
                {
                  id: String(Date.now()),
                  sender: "coco",
                  text: alertMsg,
                  timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                  isAlert: true,
                },
              ]);
            }
          }
        })
        .catch(() => {
          isDetectingRef.current = false;
        });
    }

    detectionLoopRef.current = requestAnimationFrame(runContinuousDetection);
  }, [isMirrored, speakGuidance]);

  useEffect(() => {
    if (cameraActive && cocoModelReady) {
      detectionLoopRef.current = requestAnimationFrame(runContinuousDetection);
    }
    return () => {
      if (detectionLoopRef.current) cancelAnimationFrame(detectionLoopRef.current);
    };
  }, [cameraActive, cocoModelReady, runContinuousDetection]);

  // 8. Robust Web Audio Analyser (Dynamic Real-Time Volume Meter)
  const setupAudioAnalyser = useCallback((audioStream: MediaStream) => {
    if (typeof window === "undefined") return;

    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx || audioStream.getAudioTracks().length === 0) return;

      if (audioContextRef.current) {
        try {
          audioContextRef.current.close().catch(() => {});
        } catch {}
      }

      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      analyser.smoothingTimeConstant = 0.5;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(audioStream);
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      if (volumeAnimFrameRef.current) {
        cancelAnimationFrame(volumeAnimFrameRef.current);
      }

      const checkVolume = () => {
        if (!analyserRef.current || !micActiveRef.current) {
          setAudioLevel(0);
          volumeAnimFrameRef.current = requestAnimationFrame(checkVolume);
          return;
        }

        analyserRef.current.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        const currentVol = Math.min(100, Math.round((avg / 60) * 100));
        setAudioLevel(currentVol);

        // Instant <15ms Acoustic Barge-In
        if (currentVol > 12) {
          if (conversationalStateRef.current === "reinspecting") {
            clearReinspectTimersRef.current?.();
            setConversationalState("user_speaking");
          }
          if (typeof window !== "undefined" && window.speechSynthesis) {
            if (window.speechSynthesis.speaking && !isAiSpeakingRef.current) {
              window.speechSynthesis.cancel();
              isUserSpeakingRef.current = true;
              userQueryActiveRef.current = true;
              setConversationalState("user_speaking");
            }
          }
        }

        volumeAnimFrameRef.current = requestAnimationFrame(checkVolume);
      };

      checkVolume();
    } catch (err) {
      console.warn("Audio analyser setup error:", err);
    }
  }, []);

  // 9. Independent Microphone Acquisition (Never blocked by Camera)
  const initAudioOnly = useCallback(async () => {
    if (typeof window === "undefined" || !navigator.mediaDevices?.getUserMedia) return null;
    if (audioStreamRef.current && audioStreamRef.current.active) {
      setMicActive(true);
      micActiveRef.current = true;
      return audioStreamRef.current;
    }

    try {
      const aStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      audioStreamRef.current = aStream;
      setMicActive(true);
      micActiveRef.current = true;
      setSpeechNotice(null);
      setupAudioAnalyser(aStream);
      return aStream;
    } catch (err: any) {
      console.warn("Audio acquisition error:", err?.name, err?.message);
      if (err?.name === "NotAllowedError") {
        setSpeechNotice("Microphone permission needed • Click to grant");
      }
      return null;
    }
  }, [setupAudioAnalyser]);

  // 10. Independent Camera Acquisition (Auto-retrying if in use)
  const initVideoOnly = useCallback(async () => {
    if (typeof window === "undefined" || !navigator.mediaDevices?.getUserMedia) return null;
    if (videoStreamRef.current && videoStreamRef.current.active) {
      setCameraActive(true);
      setCameraNotice(null);
      if (videoRef.current && videoRef.current.srcObject !== videoStreamRef.current) {
        videoRef.current.srcObject = videoStreamRef.current;
        videoRef.current.play().catch(() => {});
      }
      return videoStreamRef.current;
    }

    try {
      let vStream: MediaStream | null = null;
      try {
        vStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "environment",
          },
        });
      } catch {
        vStream = await navigator.mediaDevices.getUserMedia({ video: true });
      }

      videoStreamRef.current = vStream;
      setStream(vStream);
      setCameraActive(true);
      setCameraNotice(null);

      if (videoRef.current) {
        videoRef.current.srcObject = vStream;
        videoRef.current.play().catch(() => {});
      }
      return vStream;
    } catch (vErr: any) {
      setCameraActive(false);

      if (vErr?.name === "NotReadableError") {
        setCameraNotice("Camera in use by another app • Click to retry");
      } else if (vErr?.name === "NotAllowedError") {
        setCameraNotice("Camera permission needed");
      } else {
        setCameraNotice("Camera unavailable");
      }

      return null;
    }
  }, []);

  // 11. Unified Hardware Starter with AudioContext Unlock
  const unlockAudioAndStart = useCallback(async () => {
    // Un-suspend AudioContext on any interaction
    if (audioContextRef.current && audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume().catch(() => {});
    }

    // 1. Start audio first (guaranteed not blocked by camera)
    await initAudioOnly();

    // 2. Start speech recognition safely
    safeStartRecognition();

    // 3. Start video
    await initVideoOnly();
  }, [initAudioOnly, initVideoOnly, safeStartRecognition]);

  const toggleMic = useCallback(() => {
    if (micActive) {
      micActiveRef.current = false;
      setMicActive(false);
      isRecognitionActiveRef.current = false;
      try {
        recognitionRef.current?.stop();
      } catch {}
    } else {
      micActiveRef.current = true;
      setMicActive(true);
      unlockAudioAndStart();
    }
  }, [micActive, unlockAudioAndStart]);

  useEffect(() => {
    // Auto initialize on load
    unlockAudioAndStart();

    const handleUserGesture = () => {
      unlockAudioAndStart();
    };

    window.addEventListener("click", handleUserGesture);
    window.addEventListener("touchstart", handleUserGesture);
    window.addEventListener("keydown", handleUserGesture);

    return () => {
      window.removeEventListener("click", handleUserGesture);
      window.removeEventListener("touchstart", handleUserGesture);
      window.removeEventListener("keydown", handleUserGesture);
      clearReinspectTimersRef.current?.();
      if (cameraRetryTimerRef.current) clearTimeout(cameraRetryTimerRef.current);
      if (recognitionRestartTimerRef.current) clearTimeout(recognitionRestartTimerRef.current);
      if (volumeAnimFrameRef.current) cancelAnimationFrame(volumeAnimFrameRef.current);
    };
  }, [unlockAudioAndStart]);

  useEffect(() => {
    if (stream && videoRef.current && videoRef.current.srcObject !== stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }
  }, [stream]);

  const hasCenterObstacle = trackedObjects.some((o) => o.direction === "CENTER");
  const hasLeftObstacle = trackedObjects.some((o) => o.direction === "LEFT");
  const hasRightObstacle = trackedObjects.some((o) => o.direction === "RIGHT");

  return (
    <div className="min-h-screen bg-[#090a0f] text-zinc-100 flex flex-col font-sans antialiased">
      {/* Clean, Minimal Top Nav */}
      <header className="border-b border-white/[0.08] bg-[#0c0d14]/70 backdrop-blur-xl px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-white/10 border border-white/10 flex items-center justify-center text-lg">
            🦯
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-tight">Vision Guide</h1>
            <p className="text-xs text-zinc-400">Conversational Blind Assistant</p>
          </div>
        </div>

        {/* Minimal Controls */}
        <div className="flex items-center gap-2">
          {/* Mic Status & Toggle */}
          <button
            onClick={toggleMic}
            className={`px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 transition-all ${
              micActive
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20"
                : "bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20"
            }`}
            title="Toggle Microphone (speech input)"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                micActive ? "bg-emerald-400 animate-pulse" : "bg-red-400"
              }`}
            />
            <span>{micActive ? "Mic Active" : "Mic Muted"}</span>
          </button>

          <button
            onClick={() => setVoiceGuidanceEnabled(!voiceGuidanceEnabled)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
              voiceGuidanceEnabled
                ? "bg-white/10 text-white hover:bg-white/15"
                : "bg-red-500/10 text-red-400 hover:bg-red-500/20"
            }`}
            title="Toggle Voice Speech Output"
          >
            {voiceGuidanceEnabled ? "Sound ON" : "Muted"}
          </button>

          <button
            onClick={() => setIsMirrored(!isMirrored)}
            className="px-3 py-1.5 rounded-full text-xs font-medium bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-all"
            title="Flip camera"
          >
            Flip
          </button>
        </div>
      </header>

      {/* Main Clean Layout */}
      <main className="flex-1 max-w-5xl w-full mx-auto p-4 sm:p-8 flex flex-col gap-6 justify-center">
        {/* Simple Camera Viewport with Soft Rounded Corners */}
        <div className="relative rounded-3xl overflow-hidden border border-white/[0.08] bg-black shadow-2xl aspect-video w-full max-h-[460px] mx-auto flex items-center justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={() => setCameraActive(true)}
            className={`w-full h-full object-cover transition-opacity duration-300 ${
              isMirrored ? "scale-x-[-1]" : ""
            } ${!cameraActive ? "opacity-20" : "opacity-100"}`}
          />

          <canvas ref={detectionCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />

          {/* Camera Permission Inactive Fallback */}
          {!cameraActive && (
            <div
              onClick={unlockAudioAndStart}
              className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/60 backdrop-blur-sm p-6 text-center z-20 cursor-pointer"
            >
              <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-lg animate-pulse">
                📷
              </div>
              <p className="text-xs text-zinc-200 font-medium">
                {cameraNotice || "Click anywhere to start Camera & Microphone"}
              </p>
              <span className="px-5 py-2 rounded-full bg-white text-black font-semibold text-xs shadow-lg hover:bg-zinc-200 transition-all">
                {cameraNotice ? "Retrying Camera..." : "Turn On Camera & Mic"}
              </span>
            </div>
          )}

          {/* Minimal Floating 3-Way Path Indicator */}
          {cameraActive && (
            <div className="absolute top-4 inset-x-0 flex justify-center z-20 pointer-events-none">
              <div className="bg-black/60 backdrop-blur-md border border-white/10 px-4 py-1.5 rounded-full flex items-center gap-3 text-[11px] font-medium text-zinc-400 shadow-lg">
                <span className={hasLeftObstacle ? "text-amber-400 font-semibold" : "text-zinc-500"}>
                  {hasLeftObstacle ? "Left Blocked" : "Left Clear"}
                </span>
                <span className="text-zinc-700">•</span>
                <span
                  className={
                    hasCenterObstacle ? "text-red-400 font-bold" : "text-emerald-400 font-semibold"
                  }
                >
                  {hasCenterObstacle ? "Center Caution" : "Center Open"}
                </span>
                <span className="text-zinc-700">•</span>
                <span className={hasRightObstacle ? "text-amber-400 font-semibold" : "text-zinc-500"}>
                  {hasRightObstacle ? "Right Blocked" : "Right Clear"}
                </span>
              </div>
            </div>
          )}

          {/* Proactive Re-Inspection Status Overlay */}
          {conversationalState === "reinspecting" && (
            <div className="absolute bottom-4 inset-x-0 flex justify-center z-20 pointer-events-none">
              <div className="bg-amber-500/20 border border-amber-500/40 backdrop-blur-md px-5 py-2 rounded-full flex items-center gap-2.5 shadow-xl animate-pulse">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                <span className="text-xs font-semibold text-amber-200">
                  Turn body as instructed • Re-inspecting in {reinspectCountdown ?? 3}s
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Central Conversational Voice Bar (ChatGPT Style) */}
        <div className="w-full max-w-xl mx-auto flex flex-col items-center text-center gap-3">
          {/* Subtle Glowing Indicator Pill with Dynamic Mic Sound Waves */}
          <div className="flex items-center gap-3 px-5 py-2.5 rounded-full bg-white/[0.05] border border-white/[0.08] shadow-sm">
            {/* Dynamic Sound Wave Bars */}
            <div className="flex items-center gap-1 h-3.5 px-0.5" title={`Mic Volume: ${audioLevel}%`}>
              {[0.4, 0.75, 1.0, 0.8, 0.45].map((factor, i) => {
                const barHeight = Math.max(3, Math.min(16, Math.round((audioLevel / 100) * 16 * factor + 3)));
                return (
                  <span
                    key={i}
                    className={`w-1 rounded-full transition-all duration-75 ${
                      conversationalState === "user_speaking" || audioLevel > 12
                        ? "bg-amber-400"
                        : conversationalState === "reinspecting"
                        ? "bg-amber-400 animate-pulse"
                        : conversationalState === "ai_speaking"
                        ? "bg-emerald-400"
                        : micActive
                        ? "bg-zinc-400"
                        : "bg-zinc-700"
                    }`}
                    style={{ height: `${barHeight}px` }}
                  />
                );
              })}
            </div>

            <span className="text-xs font-medium text-zinc-300">
              {conversationalState === "user_speaking"
                ? "Listening to you..."
                : conversationalState === "thinking"
                ? "Inspecting view with Gemini..."
                : conversationalState === "reinspecting"
                ? `Turn as instructed • Re-inspecting in ${reinspectCountdown ?? 3}s...`
                : conversationalState === "ai_speaking"
                ? "Guiding..."
                : micActive
                ? "Listening • Speak naturally anytime"
                : "Microphone paused"}
            </span>
          </div>

          {/* Speech / Permission Notice */}
          {speechNotice && (
            <button
              onClick={unlockAudioAndStart}
              className="cursor-pointer px-4 py-1.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[11px] font-medium flex items-center gap-2 animate-pulse hover:bg-amber-500/25 transition-all"
            >
              <span>⚠️</span>
              <span>{speechNotice}</span>
            </button>
          )}

          {/* Current Spoken Transcript Snippet */}
          <p className="text-xs text-zinc-400 min-h-[18px] max-w-md truncate">
            {conversationalState === "reinspecting" ? (
              <span className="text-amber-300">Turning to inspect new angle... Speak anytime if ready</span>
            ) : currentLiveSpeech ? (
              <span className="text-zinc-200 italic">"{currentLiveSpeech}"</span>
            ) : (
              "Ask: 'Where is the door?' or 'What is in front of me?'"
            )}
          </p>

          {/* Fail-Safe Quick Query Bar & Tap-to-Speak */}
          <div className="w-full flex flex-col gap-2 pt-1">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (manualQueryText.trim()) {
                  dispatchConversationalQuery(manualQueryText.trim(), false);
                  setManualQueryText("");
                }
              }}
              className="flex items-center gap-2 bg-white/[0.04] border border-white/[0.08] rounded-2xl p-1.5 focus-within:border-white/20 transition-all"
            >
              <input
                type="text"
                value={manualQueryText}
                onChange={(e) => setManualQueryText(e.target.value)}
                placeholder="Ask query directly or tap mic..."
                className="flex-1 bg-transparent px-3 py-1 text-xs text-white placeholder-zinc-500 focus:outline-none"
              />
              <button
                type="button"
                onClick={() => {
                  unlockAudioAndStart();
                  safeStartRecognition();
                }}
                title="Force listen / Tap to speak"
                className="px-2.5 py-1.5 rounded-xl bg-white/10 hover:bg-white/15 text-zinc-300 hover:text-white transition-all text-xs flex items-center gap-1 font-medium"
              >
                <span>🎤</span>
                <span className="hidden sm:inline">Tap Mic</span>
              </button>
              <button
                type="submit"
                disabled={!manualQueryText.trim()}
                className="px-3.5 py-1.5 rounded-xl bg-white text-black font-semibold text-xs disabled:opacity-30 disabled:cursor-not-allowed hover:bg-zinc-200 transition-all"
              >
                Ask
              </button>
            </form>

            {/* Quick Suggestions Chips for Judges & Testing */}
            <div className="flex items-center justify-center gap-1.5 flex-wrap">
              {["Where is the door?", "Where is room 204?", "What is in front of me?"].map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => dispatchConversationalQuery(q, false)}
                  className="px-3 py-1 rounded-full text-[11px] bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.06] text-zinc-400 hover:text-zinc-200 transition-all"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Clean Conversational Log */}
        {navigationLogs.length > 0 && (
          <div className="w-full max-w-xl mx-auto rounded-2xl bg-white/[0.02] border border-white/[0.06] p-4 max-h-[180px] overflow-y-auto space-y-2.5 text-xs">
            {navigationLogs.slice(-4).map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col gap-0.5 ${
                  msg.sender === "user"
                    ? "text-right text-zinc-400"
                    : msg.isAlert
                    ? "text-left text-red-400 font-semibold"
                    : "text-left text-zinc-200"
                }`}
              >
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                  {msg.sender === "user" ? "You" : msg.isAlert ? "Safety Alert" : "Assistant"}
                </span>
                <p className="leading-relaxed">{msg.text}</p>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Clean, Quiet Footer */}
      <footer className="border-t border-white/[0.06] py-3 text-center text-xs text-zinc-600">
        Voice Navigation for the Visually Impaired
      </footer>
    </div>
  );
}
