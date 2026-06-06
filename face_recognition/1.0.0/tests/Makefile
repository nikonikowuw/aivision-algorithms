# 测试 Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -ldl

# 测试目标
TESTS = test_abi

# 默认目标
all: $(TESTS)

# 编译测试
test_abi: test_abi.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# 运行测试
run: test_abi
	@if [ -f ../nikoniko_detector.so ]; then \
		./test_abi ../nikoniko_detector.so; \
	else \
		echo "错误: 未找到 nikoniko_detector.so，请先编译算法包"; \
		exit 1; \
	fi

# 清理
clean:
	rm -f $(TESTS)

.PHONY: all run clean
