#include "cv.h"
#include "highgui.h"
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;

/**
 * R:4等分、G:4等分、B:4等分で4x4x4=64色に減色
 * @param[in] value 輝度値
 * @return 4等分された代表輝度値
 */
uchar decleaseColor(int value) {
    if (value < 64) {
        return 32;
    } else if (value < 128) {
        return 96;
    } else if (value < 196) {
        return 160;
    } else {
        return 224;
    }

    return 0;  // 未到達
}

/**
 * オリジナル画像のRGBからヒストグラムのビン番号を計算
 * @param[in]  red    赤の輝度（0-255）
 * @param[in]  green  緑の輝度（0-255）
 * @param[in]  blue   青の輝度（0-255）
 * @return ヒストグラムのビン番号（64色カラーインデックス）
 */
int rgb2bin(int red, int green, int blue) {
    int redNo = red / 64;
    int greenNo = green / 64;
    int blueNo = blue / 64;
    return 16 * redNo + 4 * greenNo + blueNo;
}

/**
 * ヒストグラムを計算
 * @param[in]  filename   画像ファイル名
 * @param[out] histogram  ヒストグラム
 * @return 正常終了で0、異常終了で-1
 */
int calcHistogram(char *filename, int histogram[64]) {
    // ヒストグラムを初期化
    for (int i = 0; i < 64; i++) {
        histogram[i] = 0;
    }

    // 画像のロード
    IplImage *img = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
    if (img == NULL) {
        cerr << "cannot open image: " << filename << endl;
        return -1;
    }

    // 64色に減色してヒストグラムを計算
    for (int y = 0; y < img->height; y++) {
        // y行目のデータの先頭ポインタを取得
        uchar *pin = (uchar *)(img->imageData + y * img->widthStep);
        for (int x = 0; x < img->width; x++) {
            int blue = pin[3*x+0];
            int green = pin[3*x+1];
            int red = pin[3*x+2];

            // カラー値（0-63）をカウント
            int bin = rgb2bin(red, green, blue);
            histogram[bin] += 1;
        }
    }

    cvReleaseImage(&img);

    return 0;
}

/**
 * ヒストグラムをファイルに出力
 * @param[in]  filename  出力ファイル名
 * @param[in]  histogram ヒストグラム
 * @return 正常終了で0、異常終了で-1
 */
int writeHistogram(char *filename, int histogram[64]) {
    ofstream outFile(filename);
    if (outFile.fail()) {
        cerr << "cannot open file: " << filename << endl;
        return -1;
    }

    for (int i = 0; i < 64; i++) {
        outFile << histogram[i] << endl;
    }

    outFile.close();

    return 0;
}

/**
 * メイン関数 hist.exe [入力画像ファイル名] [出力ヒストグラムファイル名]
 * @param[in]  argc
 * @param[out] argv
 * @return 正常終了で0、異常終了で-1
 */
int main(int argc, char **argv) {
    int ret;

    if (argc < 2) {
        cerr << "usage: hist.exe [image file] [hist file]" << endl;
        return -1;
    }

    char *imageFile = argv[1];
    char *histFile = argv[2];

    cout << imageFile << " -> " << histFile;

    // ヒストグラムを計算
    int histogram[64];
    ret = calcHistogram(imageFile, histogram);
    if (ret < 0) {
        cerr << "cannot calc histogram" << endl;
        return -1;
    }

    // ヒストグラムをファイルに出力
    ret = writeHistogram(histFile, histogram);
    if (ret < 0) {
        cerr << "cannot write histogram" << endl;
        return -1;
    }

    cout << " ... OK" << endl;
    return 0;
}
