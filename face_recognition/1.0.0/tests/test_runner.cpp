// 自检程序 - 加载动态库并运行 detector_self_test
#include <dlfcn.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <动态库路径>" << std::endl;
        return 2;
    }

    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        std::cerr << "加载失败: " << dlerror() << std::endl;
        return 3;
    }

    using SelfTestFn = int (*)();
    auto self_test = reinterpret_cast<SelfTestFn>(dlsym(lib, "detector_self_test"));
    if (!self_test) {
        std::cerr << "找不到 detector_self_test: " << dlerror() << std::endl;
        dlclose(lib);
        return 4;
    }

    int ret = self_test();
    dlclose(lib);
    return ret;
}
