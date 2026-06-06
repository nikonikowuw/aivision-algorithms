#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别算法包基本使用示例
"""

import json
import ctypes
import sys
import os

class FaceRecognition:
    """人脸识别算法包装器"""
    
    def __init__(self, package_path):
        """
        初始化人脸识别算法
        
        Args:
            package_path: 算法包路径
        """
        self.package_path = package_path
        self.so_path = os.path.join(package_path, "nikoniko_detector.so")
        self.handle = None
        
        # 加载动态库
        try:
            self.lib = ctypes.CDLL(self.so_path)
        except OSError as e:
            raise RuntimeError(f"无法加载动态库: {e}")
        
        # 设置函数签名
        self._setup_function_signatures()
        
        # 打印算法信息
        self._print_info()
    
    def _setup_function_signatures(self):
        """设置函数签名"""
        # detector_init
        self.lib.detector_init.argtypes = [ctypes.c_char_p]
        self.lib.detector_init.restype = ctypes.c_void_p
        
        # detector_infer
        self.lib.detector_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p
        ]
        self.lib.detector_infer.restype = ctypes.c_int
        
        # detector_free_result
        self.lib.detector_free_result.argtypes = [ctypes.c_void_p]
        self.lib.detector_free_result.restype = None
        
        # detector_destroy
        self.lib.detector_destroy.argtypes = [ctypes.c_void_p]
        self.lib.detector_destroy.restype = None
        
        # detector_version (可选)
        try:
            self.lib.detector_version.argtypes = []
            self.lib.detector_version.restype = ctypes.c_char_p
            self.has_version = True
        except AttributeError:
            self.has_version = False
        
        # detector_name (可选)
        try:
            self.lib.detector_name.argtypes = []
            self.lib.detector_name.restype = ctypes.c_char_p
            self.has_name = True
        except AttributeError:
            self.has_name = False
        
        # detector_self_test (可选)
        try:
            self.lib.detector_self_test.argtypes = []
            self.lib.detector_self_test.restype = ctypes.c_int
            self.has_self_test = True
        except AttributeError:
            self.has_self_test = False
    
    def _print_info(self):
        """打印算法信息"""
        if self.has_version:
            version = self.lib.detector_version()
            if version:
                print(f"算法版本: {version.decode('utf-8')}")
        
        if self.has_name:
            name = self.lib.detector_name()
            if name:
                print(f"算法名称: {name.decode('utf-8')}")
    
    def self_test(self):
        """
        执行自检
        
        Returns:
            bool: 自检是否通过
        """
        if not self.has_self_test:
            print("警告: 算法不支持自检")
            return True
        
        print("执行自检...")
        result = self.lib.detector_self_test()
        if result == 0:
            print("自检通过")
            return True
        else:
            print(f"自检失败: {result}")
            return False
    
    def init(self, config=None):
        """
        初始化算法
        
        Args:
            config: 配置字典，可选参数
        
        Returns:
            bool: 初始化是否成功
        """
        if config is None:
            config = {}
        
        # 设置默认配置
        config.setdefault("package_dir", self.package_path)
        config.setdefault("conf_thres", 0.5)
        config.setdefault("iou_thres", 0.45)
        config.setdefault("enable_tracker", True)
        config.setdefault("face_conf_thres", 0.6)
        config.setdefault("face_quality_thres", 0.3)
        config.setdefault("max_faces", 10)
        
        # 转换为JSON字符串
        config_json = json.dumps(config).encode('utf-8')
        
        # 初始化算法
        self.handle = self.lib.detector_init(config_json)
        
        if self.handle:
            print("算法初始化成功")
            return True
        else:
            print("算法初始化失败")
            return False
    
    def infer(self, image_data, width, height, channels=3):
        """
        执行推理
        
        Args:
            image_data: 图像数据（bytes）
            width: 图像宽度
            height: 图像高度
            channels: 图像通道数
        
        Returns:
            dict: 推理结果
        """
        if not self.handle:
            raise RuntimeError("算法未初始化，请先调用init()")
        
        # 创建硬件缓冲区描述符
        class HwBufferDesc(ctypes.Structure):
            _fields_ = [
                ("dma_fd", ctypes.c_int),
                ("size", ctypes.c_size_t),
                ("width", ctypes.c_uint32),
                ("height", ctypes.c_uint32),
                ("pixel_format", ctypes.c_uint32),
                ("dma_buf_fd", ctypes.c_int),
                ("phys_addr", ctypes.c_uint64)
            ]
        
        # 创建推理结果结构
        class InferResult(ctypes.Structure):
            _fields_ = [
                ("result_json", ctypes.c_char_p),
                ("result_json_len", ctypes.c_size_t),
                ("infer_time_us", ctypes.c_uint32),
                ("reserved", ctypes.c_int * 4)
            ]
        
        # 设置输入
        input_desc = HwBufferDesc()
        input_desc.dma_fd = -1
        input_desc.size = len(image_data)
        input_desc.width = width
        input_desc.height = height
        input_desc.pixel_format = 0  # RGB
        input_desc.dma_buf_fd = -1
        input_desc.phys_addr = 0
        
        # 创建图像数据缓冲区
        image_buffer = ctypes.create_string_buffer(image_data)
        
        # 执行推理
        result = InferResult()
        ret = self.lib.detector_infer(
            self.handle,
            ctypes.byref(input_desc),
            None,
            ctypes.byref(result)
        )
        
        if ret == 0:
            # 解析结果
            result_json = result.result_json.decode('utf-8')
            result_dict = json.loads(result_json)
            
            # 释放结果
            self.lib.detector_free_result(ctypes.byref(result))
            
            return {
                "success": True,
                "result": result_dict,
                "infer_time_us": result.infer_time_us
            }
        else:
            return {
                "success": False,
                "error": f"推理失败，错误码: {ret}"
            }
    
    def destroy(self):
        """销毁算法"""
        if self.handle:
            self.lib.detector_destroy(self.handle)
            self.handle = None
            print("算法已销毁")
    
    def __del__(self):
        """析构函数"""
        self.destroy()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <算法包路径>")
        sys.exit(1)
    
    package_path = sys.argv[1]
    
    try:
        # 创建人脸识别实例
        face_recognition = FaceRecognition(package_path)
        
        # 执行自检
        if not face_recognition.self_test():
            print("自检失败")
            sys.exit(1)
        
        # 初始化算法
        config = {
            "conf_thres": 0.5,
            "iou_thres": 0.45,
            "enable_tracker": True,
            "face_conf_thres": 0.6,
            "face_quality_thres": 0.3,
            "max_faces": 10
        }
        
        if not face_recognition.init(config):
            print("初始化失败")
            sys.exit(1)
        
        # 创建测试图像数据
        width = 640
        height = 480
        channels = 3
        image_data = bytes([128] * (width * height * channels))
        
        # 执行推理
        result = face_recognition.infer(image_data, width, height, channels)
        
        if result["success"]:
            print(f"推理成功")
            print(f"推理时间: {result['infer_time_us']} 微秒")
            print(f"结果: {json.dumps(result['result'], indent=2, ensure_ascii=False)}")
        else:
            print(f"推理失败: {result['error']}")
        
        # 销毁算法
        face_recognition.destroy()
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()