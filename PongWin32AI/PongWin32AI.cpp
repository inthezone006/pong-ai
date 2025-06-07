#include <windows.h>
#include <tensorflow/c/c_api.h>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdlib>

static void NoOpDeallocator(void* data, size_t length, void* /*arg*/) {
    // do nothing
}

// Globals for TensorFlow (we’ll load the model later)
static TF_Graph* tfGraph = nullptr;
static TF_Session* tfSession = nullptr;
static TF_Status* tfStatus = nullptr;

// Pong game state
struct Paddle { float y; };              // normalized center y ∈ [0..1]
struct Ball { float x, y, vx, vy; };  // normalized x,y ∈ [0..1], small vx,vy

static Paddle playerPaddle = { 0.5f };  // start centered vertically
static Paddle aiPaddle = { 0.5f };
static Ball   ball = { 0.5f, 0.5f, 0.01f, 0.007f };  // start center, moving down/right

static int playerScore = 0;
static int aiScore = 0;
static bool gameOver = false;

static const float PLAYER_SPEED = 0.05f;

static const int WIN_W = 800;
static const int WIN_H = 600;
#define ID_TIMER 1001

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void LoadTFModel(const char* model_dir);

void AIPaddleMove();

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nShow) {
    // 1) Register window class
    WNDCLASS wc = { };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"PongAIWin";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClass(&wc);

    // 2) Create window (client = 800×600)
    HWND hwnd = CreateWindowEx(
        0,
        L"PongAIWin",
        L"Pong (Win32 + TensorFlow AI)",
        WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WIN_W + 16, WIN_H + 39,
        nullptr, nullptr, hInst, nullptr
    );
    ShowWindow(hwnd, nShow);
    UpdateWindow(hwnd);

    // 3) Load your SavedModel (assumes folder “saved_pong_ai” next to the EXE)
    tfStatus = TF_NewStatus();
    LoadTFModel("saved_pong_ai");

    // Initialize game state
    playerPaddle.y = 0.5f;
    aiPaddle.y = 0.5f;
    ball.x = 0.5f;   ball.y = 0.5f;
    ball.vx = 0.01f; ball.vy = 0.007f;
    playerScore = 0; aiScore = 0; gameOver = false;


    // 4) Start a 60 FPS timer
    SetTimer(hwnd, ID_TIMER, 16, nullptr);

    // 5) Message loop
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 6) Cleanup TensorFlow
    if (tfSession) { TF_CloseSession(tfSession, tfStatus); TF_DeleteSession(tfSession, tfStatus); }
    if (tfGraph) { TF_DeleteGraph(tfGraph); }
    if (tfStatus) { TF_DeleteStatus(tfStatus); }

    return 0;
}

// Load the SavedModel into tfGraph/tfSession
void LoadTFModel(const char* model_dir) {
    TF_SessionOptions* opts = TF_NewSessionOptions();
    const char* tags[] = { "serve" };
    tfGraph = TF_NewGraph();
    tfSession = TF_LoadSessionFromSavedModel(
        opts, nullptr,
        model_dir,
        tags, 1,
        tfGraph, nullptr,
        tfStatus
    );
    TF_DeleteSessionOptions(opts);
    if (TF_GetCode(tfStatus) != TF_OK) {
        MessageBox(nullptr, L"Failed to load SavedModel.", L"Error", MB_OK | MB_ICONERROR);
    }
}

// Returns +1 if player scored (ball left the right edge),
//        -1 if AI scored (ball left the left edge),
//         0 otherwise.
int CheckScore() {
    if (ball.x <= 0.0f) {
        // Ball went off left → AI scores
        return -1;
    }
    if (ball.x >= 1.0f - (10.0f / static_cast<float>(WIN_W))) {
        // Ball went off right → Player scores
        return 1;
    }
    return 0;
}

// Call TensorFlow to update aiPaddle.y each frame
void AIPaddleMove() {
    if (!tfSession || !tfGraph) return;

    // 1) Prepare input [1×5] = [ball.x, ball.y, ball.vx, ball.vy, aiPaddle.y]
    float inputData[5] = {
        ball.x,
        ball.y,
        ball.vx,
        ball.vy,
        aiPaddle.y
    };
    int64_t dims[2] = { 1, 5 };
    TF_Tensor* inTensor = TF_NewTensor(
        TF_FLOAT,      // data type
        dims, 2,       // two dimensions: [1,5]
        inputData,     // pointer to your float[5] buffer
        sizeof(inputData),
        NoOpDeallocator,  // explicit no‐op deallocator
        nullptr           // deallocator arg (unused)
    );

    // 2) Locate input/output ops (names from saved_model_cli)
    TF_Output inOp = { TF_GraphOperationByName(tfGraph, "serving_default_keras_tensor"), 0 };
    TF_Output outOp = { TF_GraphOperationByName(tfGraph, "StatefulPartitionedCall_1"),     0 };
    if (!inOp.oper || !outOp.oper) {
        TF_DeleteTensor(inTensor);
        return;
    }

    // 3) Run the session
    TF_Tensor* outTensor = nullptr;
    TF_SessionRun(
        tfSession,
        nullptr,
        &inOp, &inTensor, 1,
        &outOp, &outTensor, 1,
        nullptr, 0,
        nullptr,
        tfStatus
    );
    TF_DeleteTensor(inTensor);
    if (TF_GetCode(tfStatus) != TF_OK) {
        if (outTensor) TF_DeleteTensor(outTensor);
        return;
    }

    // 4) Read the output velocity (−1..+1) and apply
    float* outVals = static_cast<float*>(TF_TensorData(outTensor));
    float vel = outVals[0];
    TF_DeleteTensor(outTensor);

    aiPaddle.y += vel * 0.04f;
    // Clamp within [0.05..0.95]
    if (aiPaddle.y < 0.05f) aiPaddle.y = 0.05f;
    if (aiPaddle.y > 0.95f) aiPaddle.y = 0.95f;
}



// Minimal window proc: just clear to black
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_KEYDOWN:
        if (!gameOver) {
            // W or Up Arrow: move up
            if (wParam == 'W' || wParam == VK_UP) {
                playerPaddle.y -= 0.02f;
                if (playerPaddle.y < 0.05f) playerPaddle.y = 0.05f;
            }
            // S or Down Arrow: move down
            else if (wParam == 'S' || wParam == VK_DOWN) {
                playerPaddle.y += 0.02f;
                if (playerPaddle.y > 0.95f) playerPaddle.y = 0.95f;
            }
        }
        return 0;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        // Fill background black
        HBRUSH brBack = CreateSolidBrush(RGB(0, 0, 0));
        RECT rc; GetClientRect(hwnd, &rc);
        FillRect(hdc, &rc, brBack);
        DeleteObject(brBack);

        // Draw center dashed line
        HPEN penDash = CreatePen(PS_DOT, 2, RGB(192, 192, 192));
        HPEN oldPen = (HPEN)SelectObject(hdc, penDash);
        for (int y = 0; y < WIN_H; y += 20) {
            MoveToEx(hdc, WIN_W / 2, y, nullptr);
            LineTo(hdc, WIN_W / 2, y + 10);
        }
        SelectObject(hdc, oldPen);
        DeleteObject(penDash);

        // Draw player paddle (left) - 20×100 px
        int px = 20;
        int py = int(playerPaddle.y * (WIN_H - 100)) + 50;
        HBRUSH brP = CreateSolidBrush(RGB(255, 255, 255));
        RECT rP = { px, py - 50, px + 20, py + 50 };
        FillRect(hdc, &rP, brP);
        DeleteObject(brP);

        // Draw AI paddle (right) - 20×100 px
        int ax = WIN_W - 40;
        int ay = int(aiPaddle.y * (WIN_H - 100)) + 50;
        HBRUSH brA = CreateSolidBrush(RGB(255, 255, 255));
        RECT rA = { ax, ay - 50, ax + 20, ay + 50 };
        FillRect(hdc, &rA, brA);
        DeleteObject(brA);

        // Draw ball - 10×10 px
        int bx = int(ball.x * (WIN_W - 10));
        int by = int(ball.y * (WIN_H - 10));
        HBRUSH brB = CreateSolidBrush(RGB(255, 255, 255));
        RECT rB = { bx, by, bx + 10, by + 10 };
        FillRect(hdc, &rB, brB);
        DeleteObject(brB);

        // Draw scores at top center
        std::wstring scoreText = L"Player: " + std::to_wstring(playerScore)
            + L"    AI: " + std::to_wstring(aiScore);
        SetTextColor(hdc, RGB(255, 255, 255));
        SetBkMode(hdc, TRANSPARENT);
        TextOut(hdc, WIN_W / 2 - 80, 10, scoreText.c_str(), (int)scoreText.size());

        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_TIMER:
        if (wParam == ID_TIMER && !gameOver) {
            // Up: W or ↑
            if (GetAsyncKeyState('W') & 0x8000 || GetAsyncKeyState(VK_UP) & 0x8000) {
                playerPaddle.y -= PLAYER_SPEED;
            }
            // Down: S or ↓
            if (GetAsyncKeyState('S') & 0x8000 || GetAsyncKeyState(VK_DOWN) & 0x8000) {
                playerPaddle.y += PLAYER_SPEED;
            }
            // Clamp
            if (playerPaddle.y < 0.05f) playerPaddle.y = 0.05f;
            if (playerPaddle.y > 0.95f) playerPaddle.y = 0.95f;
            // 1) Move the ball
            ball.x += ball.vx;
            ball.y += ball.vy;

            // 2) Bounce off top/bottom
            if (ball.y <= 0.0f) {
                ball.y = 0.0f;
                ball.vy = -ball.vy;
            }
            else if (ball.y >= 1.0f - (10.0f / static_cast<float>(WIN_H))) {
                ball.y = 1.0f - (10.0f / static_cast<float>(WIN_H));
                ball.vy = -ball.vy;
            }

            // 3) Collision with player paddle
            RECT playerRect = {
                20,
                static_cast<int>(playerPaddle.y * (WIN_H - 100)) + 50 - 50,
                20 + 20,
                static_cast<int>(playerPaddle.y * (WIN_H - 100)) + 50 + 50
            };
            RECT ballRect = {
                static_cast<int>(ball.x * (WIN_W - 10)),
                static_cast<int>(ball.y * (WIN_H - 10)),
                static_cast<int>(ball.x * (WIN_W - 10)) + 10,
                static_cast<int>(ball.y * (WIN_H - 10)) + 10
            };
            if (IntersectRect(&playerRect, &playerRect, &ballRect)) {
                ball.vx = fabsf(ball.vx);
            }

            // 4) Collision with AI paddle
            RECT aiRect = {
                WIN_W - 40,
                static_cast<int>(aiPaddle.y * (WIN_H - 100)) + 50 - 50,
                WIN_W - 40 + 20,
                static_cast<int>(aiPaddle.y * (WIN_H - 100)) + 50 + 50
            };
            if (IntersectRect(&aiRect, &aiRect, &ballRect)) {
                ball.vx = -fabsf(ball.vx);
            }

            // 5) Check for scoring
            int scored = CheckScore();
            if (scored != 0) {
                if (scored == 1)       ++playerScore;
                else /* scored==-1 */ ++aiScore;

                // Reset ball to center with random direction
                ball.x = 0.5f;
                ball.y = 0.5f;
                ball.vx = (rand() % 2 ? 1.0f : -1.0f) * 0.01f;
                ball.vy = (rand() % 2 ? 1.0f : -1.0f) * 0.007f;

                // If someone reached 10, end the game
                if (playerScore >= 10 || aiScore >= 10) {
                    gameOver = true;
                }
            }
            if (!gameOver) {
                AIPaddleMove();
            }
            // 6) Redraw
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;
    case WM_DESTROY:
        KillTimer(hwnd, ID_TIMER);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}
