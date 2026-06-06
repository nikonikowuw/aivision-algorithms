# 示例代码编译

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -ldl

# 目标
TARGETS = basic_usage

# 源文件
SRCS = basic_usage.cpp

# 默认目标
all: $(TARGETS)

# 编译示例
basic_usage: basic_usage.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# 清理
clean:
	rm -f $(TARGETS)

# 安装
install: all
	@echo "示例代码编译完成"
	@echo "运行示例: ./basic_usage <算法包路径>"

.PHONY: all clean install