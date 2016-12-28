#include "cv.h"
#include "highgui.h"
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;

/**
 * R:4�����AG:4�����AB:4������4x4x4=64�F�Ɍ��F
 * @param[in] value �P�x�l
 * @return 4�������ꂽ��\�P�x�l
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

    return 0;  // �����B
}

/**
 * �I���W�i���摜��RGB����q�X�g�O�����̃r���ԍ����v�Z
 * @param[in]  red    �Ԃ̋P�x�i0-255�j
 * @param[in]  green  �΂̋P�x�i0-255�j
 * @param[in]  blue   �̋P�x�i0-255�j
 * @return �q�X�g�O�����̃r���ԍ��i64�F�J���[�C���f�b�N�X�j
 */
int rgb2bin(int red, int green, int blue) {
    int redNo = red / 64;
    int greenNo = green / 64;
    int blueNo = blue / 64;
    return 16 * redNo + 4 * greenNo + blueNo;
}

/**
 * �q�X�g�O�������v�Z
 * @param[in]  filename   �摜�t�@�C����
 * @param[out] histogram  �q�X�g�O����
 * @return ����I����0�A�ُ�I����-1
 */
int calcHistogram(char *filename, int histogram[64]) {
    // �q�X�g�O������������
    for (int i = 0; i < 64; i++) {
        histogram[i] = 0;
    }

    // �摜�̃��[�h
    IplImage *img = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
    if (img == NULL) {
        cerr << "cannot open image: " << filename << endl;
        return -1;
    }

    // 64�F�Ɍ��F���ăq�X�g�O�������v�Z
    for (int y = 0; y < img->height; y++) {
        // y�s�ڂ̃f�[�^�̐擪�|�C���^���擾
        uchar *pin = (uchar *)(img->imageData + y * img->widthStep);
        for (int x = 0; x < img->width; x++) {
            int blue = pin[3*x+0];
            int green = pin[3*x+1];
            int red = pin[3*x+2];

            // �J���[�l�i0-63�j���J�E���g
            int bin = rgb2bin(red, green, blue);
            histogram[bin] += 1;
        }
    }

    cvReleaseImage(&img);

    return 0;
}

/**
 * �q�X�g�O�������t�@�C���ɏo��
 * @param[in]  filename  �o�̓t�@�C����
 * @param[in]  histogram �q�X�g�O����
 * @return ����I����0�A�ُ�I����-1
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
 * ���C���֐� hist.exe [���͉摜�t�@�C����] [�o�̓q�X�g�O�����t�@�C����]
 * @param[in]  argc
 * @param[out] argv
 * @return ����I����0�A�ُ�I����-1
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

    // �q�X�g�O�������v�Z
    int histogram[64];
    ret = calcHistogram(imageFile, histogram);
    if (ret < 0) {
        cerr << "cannot calc histogram" << endl;
        return -1;
    }

    // �q�X�g�O�������t�@�C���ɏo��
    ret = writeHistogram(histFile, histogram);
    if (ret < 0) {
        cerr << "cannot write histogram" << endl;
        return -1;
    }

    cout << " ... OK" << endl;
    return 0;
}
