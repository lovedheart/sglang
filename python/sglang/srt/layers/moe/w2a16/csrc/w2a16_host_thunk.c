/* ig_host_thunk — exported trampoline so cudaLaunchHostFunc can call back
 * into Python. The CUDA host-fn ABI is void(void*); we keep one registered
 * Python callable (CFUNCTYPE(void, c_void_p)) and forward userData to it. */
typedef void (*py_cb_t)(void *);
static py_cb_t g_cb = 0;

void ig_set_cb(void *cb) { g_cb = (py_cb_t)cb; }

void ig_host_thunk(void *userData) {
    if (g_cb) g_cb(userData);
}
