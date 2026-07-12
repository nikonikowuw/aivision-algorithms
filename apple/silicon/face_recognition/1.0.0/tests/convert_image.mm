/**
 * @file convert_image.mm
 * @brief 将 JPEG 图像转换为原始 BGR24 格式供 e2e 测试使用
 *        Convert JPEG image to raw BGR24 format for e2e test
 *
 * 使用 CGDataProviderCopyData 直接从 ImageIO 解码的 CGImage 读取原始像素数据，
 * 绕过 CoreGraphics 色彩空间转换，输出与 OpenCV imread 兼容的 BGR24。
 *
 * Output format: width(uint32_t) + height(uint32_t) + BGR24 pixels
 */
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#include <cstdio>
#include <cstdint>

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSString *inputPath = @"testimage.jpg";
        NSString *outputPath = @"/tmp/testimage_raw.bgr";

        if (argc > 1) inputPath = [NSString stringWithUTF8String:argv[1]];
        if (argc > 2) outputPath = [NSString stringWithUTF8String:argv[2]];

        // 1. 通过 ImageIO 解码 JPEG
        NSURL *url = [NSURL fileURLWithPath:inputPath];
        CGImageSourceRef source = CGImageSourceCreateWithURL((CFURLRef)url, NULL);
        if (!source) {
            fprintf(stderr, "Failed to open image: %s\n", [inputPath UTF8String]);
            return 1;
        }

        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!image) {
            fprintf(stderr, "Failed to decode image\n");
            return 1;
        }

        size_t width = CGImageGetWidth(image);
        size_t height = CGImageGetHeight(image);
        size_t bpp = CGImageGetBitsPerPixel(image);
        CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(image);

        // 2. 直接从 CGImage 的数据提供者读取原始像素数据（跳过色彩管理）
        //    Get raw pixel data directly from the CGImage data provider
        CGDataProviderRef provider = CGImageGetDataProvider(image);
        if (!provider) {
            fprintf(stderr, "Failed to get data provider\n");
            CGImageRelease(image);
            return 1;
        }

        CFDataRef rawData = CGDataProviderCopyData(provider);
        if (!rawData) {
            fprintf(stderr, "Failed to copy raw data\n");
            CGImageRelease(image);
            return 1;
        }

        const uint8_t* pixels = CFDataGetBytePtr(rawData);
        size_t bytesPerRow = CGImageGetBytesPerRow(image);

        // 3. 确定像素格式：根据 AlphaInfo 和 ByteOrder 解码
        //    Determine pixel format from AlphaInfo and ByteOrder
        CGImageAlphaInfo alphaInfo = static_cast<CGImageAlphaInfo>(bitmapInfo & kCGBitmapAlphaInfoMask);
        bool hasAlpha = (alphaInfo != kCGImageAlphaNone && alphaInfo != kCGImageAlphaNoneSkipFirst && alphaInfo != kCGImageAlphaNoneSkipLast);
        bool alphaFirst = (alphaInfo == kCGImageAlphaPremultipliedFirst || alphaInfo == kCGImageAlphaFirst || alphaInfo == kCGImageAlphaNoneSkipFirst);
        bool littleEndian = (bitmapInfo & kCGBitmapByteOrderMask) == kCGBitmapByteOrder32Little;

        printf("  Image: %zux%zu, bpp=%zu, hasAlpha=%d, alphaFirst=%d, littleEndian=%d\n",
               width, height, bpp, hasAlpha, alphaFirst, littleEndian);

        size_t samplesPerPixel = bpp / 8;  // Typically 4 (RGBA) or 3 (RGB)

        // 4. 写入输出文件：BGR24
        //    Write output file: BGR24
        FILE* f = fopen([outputPath UTF8String], "wb");
        if (!f) {
            fprintf(stderr, "Failed to create output file\n");
            CFRelease(rawData);
            CGImageRelease(image);
            return 1;
        }

        uint32_t w = static_cast<uint32_t>(width);
        uint32_t h = static_cast<uint32_t>(height);
        fwrite(&w, sizeof(w), 1, f);
        fwrite(&h, sizeof(h), 1, f);

        // 遍历像素，根据格式提取 R, G, B
        for (size_t y = 0; y < height; y++) {
            const uint8_t* row = pixels + y * bytesPerRow;
            for (size_t x = 0; x < width; x++) {
                uint8_t r, g, b;
                
                if (littleEndian) {
                    if (alphaFirst) {
                        // Little-endian, alpha first: [A, B, G, R] or [X, B, G, R]
                        b = row[x * samplesPerPixel + 1];
                        g = row[x * samplesPerPixel + 2];
                        r = row[x * samplesPerPixel + 3];
                    } else {
                        // Little-endian, alpha last: [B, G, R, A] or [B, G, R, X]
                        b = row[x * samplesPerPixel + 0];
                        g = row[x * samplesPerPixel + 1];
                        r = row[x * samplesPerPixel + 2];
                    }
                } else {
                    if (alphaFirst) {
                        // Big-endian, alpha first: [A, R, G, B] or [X, R, G, B]
                        r = row[x * samplesPerPixel + 1];
                        g = row[x * samplesPerPixel + 2];
                        b = row[x * samplesPerPixel + 3];
                    } else {
                        // Big-endian, alpha last: [R, G, B, A] or [R, G, B, X]
                        r = row[x * samplesPerPixel + 0];
                        g = row[x * samplesPerPixel + 1];
                        b = row[x * samplesPerPixel + 2];
                    }
                }

                // 处理 premultiplied alpha: 需要 un-premultiply
                // 对于 alpha=255 的像素，premultiplication 没有影响
                if (hasAlpha) {
                    uint8_t a = 255;
                    if (alphaFirst && littleEndian) {
                        a = row[x * samplesPerPixel + 0];
                    } else if (!alphaFirst && littleEndian) {
                        a = row[x * samplesPerPixel + 3];
                    } else if (alphaFirst && !littleEndian) {
                        a = row[x * samplesPerPixel + 0];
                    } else {
                        a = row[x * samplesPerPixel + 3];
                    }
                    
                    // 不处理 premultiplied alpha（这个 JPEG 是 opaque 的）
                    // Skip premultiplied alpha handling (this JPEG is opaque)
                    (void)a;
                }

                fputc(b, f);
                fputc(g, f);
                fputc(r, f);
            }
        }

        fclose(f);
        CFRelease(rawData);
        CGImageRelease(image);

        printf("  Converted %s (%zux%zu) to raw BGR: %s\n",
               [inputPath UTF8String], width, height, [outputPath UTF8String]);
    }
    return 0;
}
