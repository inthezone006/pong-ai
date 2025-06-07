#include <windows.h>
#include <tensorflow/c/c_api.h>
#include <string>

// Helper to convert char* to std::wstring
std::wstring to_wstring_utf8(const char* s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], len);
    return w;
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    // Create status and session options
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* opts = TF_NewSessionOptions();

    // Prepare graph and load the SavedModel
    TF_Graph* graph = TF_NewGraph();
    const char* tags[] = { "serve" };
    TF_Session* session = TF_LoadSessionFromSavedModel(
        opts,
        nullptr,                     // run options
        "saved_pong_ai",             // relative path to your model folder
        tags, 1,
        graph,
        nullptr,                     // meta graph (unused)
        status
    );

    // Check if load succeeded
    if (TF_GetCode(status) != TF_OK) {
        std::wstring msg = to_wstring_utf8(TF_Message(status));
        MessageBox(nullptr, msg.c_str(), L"Load Error", MB_OK | MB_ICONERROR);
    }
    else {
        MessageBox(nullptr, L"Model loaded successfully!", L"Success", MB_OK);
    }

    // Cleanup
    if (session) {
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
    }
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(opts);
    TF_DeleteStatus(status);

    return 0;
}
