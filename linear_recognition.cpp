#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <map>

using namespace std;

const int DIM = 128;
const double DIST_THRESHOLD = 0.25;
const double VOTE_THRESHOLD = 50;
const int SURF_PARAM = 400;

const char* IMAGE_DIR = "../dataset/caltech101_10";
const char* OBJID_FILE = "../dataset/object_caltech101_10.txt";
const char* DESC_FILE = "../dataset/description_caltech101_10.txt";

// プロトタイプ宣言
bool loadObjectId(const char *filename, map<int, string>& id2name);
bool loadDescription(const char *filename, vector<int> &labels, vector<int> &laplacians, CvMat* &objMat);
double euclideanDistance(float *vec, float *mvec, int length);
int searchNN(float *vec, int lap, vector<int> &labels, vector<int> &laplacians, CvMat* objMat);

int main(int argc, char** argv) {
    double tt = (double)cvGetTickCount();

    // 物体ID->物体ファイル名のハッシュを作成
    cout << "物体ID->物体名のハッシュを作成します ... " << flush;
    map<int, string> id2name;
    if (!loadObjectId(OBJID_FILE, id2name)) {
        cerr << "cannot load object id file" << endl;
        return 1;
    }
    cout << "OK" << endl;

    // キーポイントの特徴ベクトルをobjMat行列にロード
    cout << "物体モデルデータベースをロードします ... " << flush;
    vector<int> labels;     // キーポイントのラベル（objMatに対応）
    vector<int> laplacians;  // キーポイントのラプラシアン
    CvMat* objMat;           // 各行が物体のキーポイントの特徴ベクトル
    if (!loadDescription(DESC_FILE, labels, laplacians, objMat)) {
        cerr << "cannot load description file" << endl;
        return 1;
    }
    cout << "OK" << endl;

    cout << "物体モデルデータベースの物体数: " << id2name.size() << endl;
    cout << "データベース中のキーポイント数: " << objMat->rows << endl;
    tt = (double)cvGetTickCount() - tt;
    cout << "Loading Models Time = " << tt / (cvGetTickFrequency() * 1000.0) << "ms" << endl;

    while (1) {
        // クエリファイルの入力
        char input[1024];
        cout << "query? > ";
        cin >> input;

        char queryFile[1024];
        snprintf(queryFile, sizeof queryFile, "%s/%s", IMAGE_DIR, input);

        cout << queryFile << endl;

        tt = (double)cvGetTickCount();

        // クエリ画像をロード
        IplImage *queryImage = cvLoadImage(queryFile, CV_LOAD_IMAGE_GRAYSCALE);
        if (queryImage == NULL) {
            cerr << "cannot load image file: " << queryFile << endl;
            continue;
        }

        // クエリからSURF特徴量を抽出
        CvSeq *queryKeypoints = 0;
        CvSeq *queryDescriptors = 0;
        CvMemStorage *storage = cvCreateMemStorage(0);
        CvSURFParams params = cvSURFParams(SURF_PARAM, 1);
        cvExtractSURF(queryImage, 0, &queryKeypoints, &queryDescriptors, storage, params);
        cout << "クエリのキーポイント数: " << queryKeypoints->total << endl;

        // クエリの各キーポイントの1-NNの物体IDを検索して投票
        int numObjects = (int)id2name.size();  // データベース中の物体数
        int votes[numObjects];  // 各物体の集めた得票数
        for (int i = 0; i < numObjects; i++) {
            votes[i] = 0;
        }
        for (int i = 0; i < queryDescriptors->total; i++) {
            CvSURFPoint *p = (CvSURFPoint *)cvGetSeqElem(queryKeypoints, i);
            float *vec = (float *)cvGetSeqElem(queryDescriptors, i);
            int lap = p->laplacian;
            int nnId = searchNN(vec, lap, labels, laplacians, objMat);
            votes[nnId]++;
        }

        // 投票数が最大の物体IDを求める
        int maxId = -1;
        int maxVal = -1;
        for (int i = 0; i < numObjects; i++) {
            if (votes[i] > maxVal) {
                maxId = i;
                maxVal = votes[i];
            }
        }

        // 物体IDを物体ファイル名に変換
        string name = id2name[maxId];
        cout << "識別結果: " << name << endl;

        tt = (double)cvGetTickCount() - tt;
        cout << "Recognition Time = " << tt / (cvGetTickFrequency() * 1000.0) << "ms" << endl;

        // 後始末
        cvReleaseImage(&queryImage);
        cvClearSeq(queryKeypoints);
        cvClearSeq(queryDescriptors);
        cvReleaseMemStorage(&storage);
        cvDestroyAllWindows();
    }

    // 後始末
    cvReleaseMat(&objMat);

    return 0;
}

/**
 * 物体ID->物体名のmapを作成して返す
 *
 * @param[in]  filename  物体ID->物体名の対応を格納したファイル
 * @param[out] id2name   物体ID->物体名のmap
 *
 * @return 成功ならtrue、失敗ならfalse
 */
bool loadObjectId(const char *filename, map<int, string>& id2name) {
    // 物体IDと物体名を格納したファイルを開く
    ifstream objFile(filename);
    if (objFile.fail()) {
        cerr << "cannot open file: " << filename << endl;
        return false;
    }

    // 1行ずつ読み込み、物体ID->物体名のmapを作成
    string line;
    while (getline(objFile, line, '\n')) {
        // タブで分割した文字列をldataへ格納
        vector<string> ldata;
        istringstream ss(line);
        string s;
        while (getline(ss, s, '\t')) {
            ldata.push_back(s);
        }

        // 物体IDと物体名を抽出してmapへ格納
        int objId = atol(ldata[0].c_str());
        string objName = ldata[1];
        id2name.insert(map<int, string>::value_type(objId, objName));
    }

    // 後始末
    objFile.close();

    return true;
}

/**
 * キーポイントのラベル（抽出元の物体ID）とラプラシアンと特徴ベクトルをロードしlabelsとobjMatへ格納
 *
 * @param[in]  filename         特徴ベクトルを格納したファイル
 * @param[out] labels           特徴ベクトル抽出元の物体ID
 * @param[out] laplacianss      特徴ベクトルのラプラシアン
 * @param[out] objMat           特徴量を格納した行列（各行に1つの特徴ベクトル）
 *
 * @return 成功ならtrue、失敗ならfalse
 */
bool loadDescription(const char *filename, vector<int> &labels, vector<int> &laplacians, CvMat* &objMat) {
    // 物体IDと特徴ベクトルを格納したファイルを開く
    ifstream descFile(filename);
    if (descFile.fail()) {
        cerr << "cannot open file: " << filename << endl;
        return false;
    }

    // 行列のサイズを決定するためキーポイントの総数をカウント
    int numKeypoints = 0;
    string line;
    while (getline(descFile, line, '\n')) {
        numKeypoints++;
    }
    objMat = cvCreateMat(numKeypoints, DIM, CV_32FC1);

    // ファイルポインタを先頭に戻す
    descFile.clear();
    descFile.seekg(0);

    // データを読み込んで行列へ格納
    int cur = 0;
    while (getline(descFile, line, '\n')) {
        // タブで分割した文字列をldataへ格納
        vector<string> ldata;
        istringstream ss(line);
        string s;
        while (getline(ss, s, '\t')) {
            ldata.push_back(s);
        }
        // 物体IDを取り出して特徴ベクトルのラベルとする
        int objId = atol(ldata[0].c_str());
        labels.push_back(objId);
        // ラプラシアンを取り出して格納
        int laplacian = atoi(ldata[1].c_str());
        laplacians.push_back(laplacian);
        // DIM次元ベクトルの要素を行列へ格納
        for (int j = 0; j < DIM; j++) {
            float val = atof(ldata[j+2].c_str());  // 特徴ベクトルはldata[2]から
            CV_MAT_ELEM(*objMat, float, cur, j) = val;
        }
        cur++;
    }

    descFile.close();

    return true;
}

/**
 * 2つのベクトルのユークリッド距離を計算して返す
 *
 * @param[in] vec     ベクトル1の配列
 * @param[in] mvec    ベクトル2の配列
 * @param[in] length  ベクトルの長さ
 *
 * @return ユークリッド距離
 */
double euclideanDistance(float *vec, float *mvec, int length) {
    double dist = 0.0;

    for (int i = 0; i < length; i++) {
        dist += (vec[i] - mvec[i]) * (vec[i] - mvec[i]);
    }
    dist = sqrt(dist);

    return dist;
}

/**
 * クエリのキーポイントの1-NNキーポイントを物体モデルデータベースから探してその物体IDを返す
 *
 * @param[in] vec          クエリキーポイントの特徴ベクトル
 * @param[in] lap          クエリキーポイントのラプラシアン
 * @param[in] labels       物体モデルデータベースの各キーポイントの物体ID
 * @param[in] laplacians   物体モデルデータベースの各キーポイントのラプラシアン
 * @param[in] objMat       物体モデルデータベースの各キーポイントの特徴ベクトル
 *
 * @return 指定したキーポイントにもっとも近いキーポイントの物体ID
 */
int searchNN(float *vec, int lap, vector<int> &labels, vector<int> &laplacians, CvMat* objMat) {
    int neighborObj = -1;
    double minDist = 1e6;
    for (int i = 0; i < objMat->rows; i++) {
        // クエリのキーポイントとラプラシアンが異なるキーポイントは無視
        if (lap != laplacians[i]) {
            continue;
        }
        // i行目の先頭データのポインタ（ここから128個がi行目の特徴ベクトル）
        float *mvec = (float *)(objMat->data.fl + i * DIM);
        double d = euclideanDistance(vec, mvec, DIM);
        if (d < minDist) {
            neighborObj = labels[i];  // NNキーポイントの物体IDを更新
            minDist = d;           // 最小の距離を更新
        }
    }
    return neighborObj;
}
