# Sign Language Detection Pipeline - Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Camera
    participant FrameBuffer
    participant MediaPipe
    participant SignDetector
    participant WordDetector
    participant Display
    
    %% Initialization
    User->>FrameBuffer: Initialize
    User->>SignDetector: Load Model (YES/NO)
    User->>WordDetector: Load Model (GRU/LSTM)
    User->>Camera: Start Camera
    
    %% Main Loop
    loop Every Frame
        Camera->>FrameBuffer: Capture Frame
        FrameBuffer->>MediaPipe: Extract Landmarks
        MediaPipe-->>FrameBuffer: Hand/Pose Keypoints
        FrameBuffer->>FrameBuffer: Add to Buffer
        
        alt Buffer Ready
            FrameBuffer->>SignDetector: Get Sequence
            SignDetector->>SignDetector: Predict YES/NO
            
            alt Sign Detected (YES)
                SignDetector->>WordDetector: Forward Sequence
                WordDetector->>WordDetector: Classify Word
                WordDetector-->>Display: Word + Confidence
            else No Sign (NO)
                SignDetector-->>Display: No Action
            end
        end
        
        FrameBuffer->>Display: Render Frame
        Display->>User: Show Result
    end
    
    User->>Camera: Stop
```

## Pipeline Components

- **FrameBuffer**: Manages sliding window of frames and landmarks
- **MediaPipe**: Extracts hand/pose keypoints (126 or 227 dims)
- **SignDetector**: Binary classifier for YES/NO detection (LSTM/Transformer)
- **WordDetector**: Multi-class word classifier (GRU/LSTM)

## Pipeline Flow

```
Camera → FrameBuffer → MediaPipe → Landmark Buffer
                                         ↓
                                   SignDetector (YES/NO)
                                         ↓
                              [If YES] → WordDetector → Display
```
