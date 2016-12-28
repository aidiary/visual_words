#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <dirent.h>

using namespace std;

const char* IMAGE_DIR = "caltech10";
const int DIM = 128;
const int SURF_PARAM = 400;
const int MAX_CLUSTER = 500;  // クラスタ数 = Visual Wordsの次元数

/**
 * 画像ファイルからSURF特徴量を抽出する
 * @param[in]  filename            画像ファイル名
 * @param[out] imageKeypoints      キーポイント
 * @param[out] imageDescriptors    各キーポイントのSURF特徴量
 * @param[out] storage             メモリ領域
 * @return 成功なら0、失敗なら1
 */
int extractSURF(const char* filename, CvSeq** keypoints, CvSeq** descriptors, CvMemStorage** storage) {
    // グレースケールで画像をロードする
    IplImage* img = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (img == NULL) {
        cerr << "cannot load image: " << filename << endl;
        return 1;
    }

    *storage = cvCreateMemStorage(0);
    CvSURFParams params = cvSURFParams(SURF_PARAM, 1);
    cvExtractSURF(img, 0, keypoints, descriptors, *storage, params);

    return 0;
}

/**
 * IMAGE_DIRにある全画像から局所特徴量を抽出し行列へ格納する
 * @param[out]   samples    局所特徴量の行列
 * @param[out]   data       samplesのデータ領域
 * @return 成功なら0、失敗なら1
 */
int loadDescriptors(CvMat& samples, vector<float>& data) {
    // IMAGE_DIRを開く
    DIR* dp;
    if ((dp = opendir(IMAGE_DIR)) == NULL) {
        cerr << "cannot open directory: " << IMAGE_DIR << endl;
        return 1;
    }

    // IMAGE_DIRの画像ファイル名を走査
    struct dirent* entry;
    while ((entry = readdir(dp)) != NULL) {
        char* filename = entry->d_name;
        if (strcmp(filename, ".") == 0 || strcmp(filename, "..") == 0) {
            continue;
        }

        // パス名に変換
        // XXX.jpg -> IMAGE_DIR/XXX.jpg
        char filepath[256];
        snprintf(filepath, sizeof filepath, "%s/%s", IMAGE_DIR, filename);

        // SURFを抽出
        CvSeq* keypoints = NULL;
        CvSeq* descriptors = NULL;
        CvMemStorage* storage = NULL;
        int ret = extractSURF(filepath, &keypoints, &descriptors, &storage);
        if (ret != 0) {
            cerr << "error in extractSURF" << endl;
            return 1;
        }

        // ファイル名と局所特徴点の数を表示
        cout << filepath << "\t" << descriptors->total << endl;

        // 特徴量を構造化せずにdataへ追加
        for (int i = 0; i < descriptors->total; i++) {
            float* d = (float*)cvGetSeqElem(descriptors, i);  // 128次元ベクトル
            for (int j = 0; j < DIM; j++) {
                data.push_back(d[j]);
            }
        }

        cvClearSeq(keypoints);
        cvClearSeq(descriptors);
        cvReleaseMemStorage(&storage);
    }

    // dataをCvMat形式に変換
    // CvMatはdataを参照するためdataは解放されないので注意
    int rows = data.size() / DIM;  // CvMatの行数（=DIM次元特徴ベクトルの本数）
    cvInitMatHeader(&samples, rows, DIM, CV_32FC1, &data[0]);

    return 0;
}

/**
 * IMAEG_DIRの全画像をヒストグラムに変換して出力
 * 各画像の各局所特徴量を一番近いVisual Wordsに投票してヒストグラムを作成する
 * @param[in]   visualWords     Visual Words
 * @return 成功なら0、失敗なら1
 */
int calcHistograms(CvMat* visualWords) {
    // 一番近いVisual Wordsを高速検索できるようにvisualWordsをkd-treeでインデキシング
    CvFeatureTree* ft = cvCreateKDTree(visualWords);

    // 各画像のヒストグラムを出力するファイルを開く
    fstream fout;
    fout.open("histograms.txt", ios::out);
    if (!fout.is_open()) {
        cerr << "cannot open file: histograms.txt" << endl;
        return 1;
    }

    // IMAGE_DIRの各画像をヒストグラムに変換
    DIR* dp;
    if ((dp = opendir(IMAGE_DIR)) == NULL) {
        cerr << "cannot open directory: " << IMAGE_DIR << endl;
        return 1;
    }

    struct dirent* entry;
    while ((entry = readdir(dp)) != NULL) {
        char* filename = entry->d_name;
        if (strcmp(filename, ".") == 0 || strcmp(filename, "..") == 0) {
            continue;
        }

        char filepath[256];
        snprintf(filepath, sizeof filepath, "%s/%s", IMAGE_DIR, filename);

        // ヒストグラムを初期化
        int* histogram = new int[visualWords->rows];
        for (int i = 0; i < visualWords->rows; i++) {
            histogram[i] = 0;
        }

        // SURFを抽出
        CvSeq* keypoints = NULL;
        CvSeq* descriptors = NULL;
        CvMemStorage* storage = NULL;
        int ret = extractSURF(filepath, &keypoints, &descriptors, &storage);
        if (ret != 0) {
            cerr << "error in extractSURF" << endl;
            return 1;
        }

        // kd-treeで高速検索できるように特徴ベクトルをCvMatに展開
        CvMat* mat = cvCreateMat(descriptors->total, DIM, CV_32FC1);
        CvSeqReader reader;
        float* ptr = mat->data.fl;
        cvStartReadSeq(descriptors, &reader);
        for (int i = 0; i < descriptors->total; i++) {
            float* desc = (float*)reader.ptr;
            CV_NEXT_SEQ_ELEM(reader.seq->elem_size, reader);
            memcpy(ptr, desc, DIM*sizeof(float));
            ptr += DIM;
        }

        // 各局所特徴点についてもっとも類似したVisual Wordsを見つけて投票
        int k = 1;  // 1-NN
        CvMat* indices = cvCreateMat(keypoints->total, k, CV_32SC1);  // もっとも近いVisual Wordsのインデックス
        CvMat* dists = cvCreateMat(keypoints->total, k, CV_64FC1);    // その距離
        cvFindFeatures(ft, mat, indices, dists, k, 250);
        for (int i = 0; i < indices->rows; i++) {
            int idx = CV_MAT_ELEM(*indices, int, i, 0);
            histogram[idx] += 1;
        }

        // ヒストグラムをファイルに出力
        fout << filepath << "\t";
        for (int i = 0; i < visualWords->rows; i++) {
            fout << float(histogram[i]) / float(descriptors->total) << "\t";
        }
        fout << endl;

        // 後始末
        delete[] histogram;
        cvClearSeq(keypoints);
        cvClearSeq(descriptors);
        cvReleaseMemStorage(&storage);
        cvReleaseMat(&mat);
        cvReleaseMat(&indices);
        cvReleaseMat(&dists);
    }

    fout.close();
    cvReleaseFeatureTree(ft);

    return 0;
}

int main() {
    int ret;

    // IMAGE_DIRの各画像から局所特徴量を抽出
    cout << "Load Descriptors ..." << endl;
    CvMat samples;
    vector<float> data;
    ret = loadDescriptors(samples, data);

    // 局所特徴量をクラスタリングして各クラスタのセントロイドを計算
    cout << "Clustering ..." << endl;
    CvMat* labels = cvCreateMat(samples.rows, 1, CV_32S);        // 各サンプル点が割り当てられたクラスタのラベル
    CvMat* centroids = cvCreateMat(MAX_CLUSTER, DIM, CV_32FC1);  // 各クラスタの中心（セントロイド） DIM次元ベクトル
    cvKMeans2(&samples, MAX_CLUSTER, labels, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 1, 0, 0, centroids, 0);
    cvReleaseMat(&labels);  // ラベルは使わない

    // 各画像をVisual Wordsのヒストグラムに変換する
    // 各クラスターの中心ベクトル、centroidsがそれぞれVisual Wordsになる
    cout << "Calc Histograms ..." << endl;
    calcHistograms(centroids);
    cvReleaseMat(&centroids);

    return 0;
}
