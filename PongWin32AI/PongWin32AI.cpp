#include <windows.h>
#include <tensorflow/c/c_api.h>

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    // Display the linked TensorFlow version
    const char* version = TF_Version();
    MessageBoxA(nullptr, version, "TensorFlow Version", MB_OK);
    return 0;
}
