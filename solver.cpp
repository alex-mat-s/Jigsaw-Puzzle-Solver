#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
using namespace std;


const int W = 1200;
const int H = 900;
const int CHANNEL_NUM = 3;
const string IMG_EXTENSION = ".ppm";
const int IMG_NAME_LENGTH = 4;
const int MAX_VALUE = 255;

void readImage(ifstream& file, vector<vector<vector<int>>>& image){
    int h, w, r, g, b;
    string line;
// scan through header
    for (int i = 0; i < 3; i++){
        if (i == 1){
//          read image dimensions from second header line
            file >> w >> h;
            image.resize(h, vector<vector<int>>(w, vector<int>(CHANNEL_NUM)));
        }
        getline(file, line);
    }

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            file >> r >> g >> b;
            image[y][x][0] = r;
            image[y][x][1] = g;
            image[y][x][2] = b;
        }
    }
}


void writeImage(const vector<vector<vector<int>>>& image, const string& output_path)
{
    ofstream out(output_path);
    int h = image.size();
    int w = image[0].size();
    // write header
    out << "P3" << endl;
    out << w << " " << h << endl;
    out << MAX_VALUE << endl;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            out << image[y][x][0] << " ";
            out << image[y][x][1] << " ";
            out << image[y][x][2] << " ";
        }
    }
}


// the mean distribution of gradients
// разность между соседними столбцами пикселей в одной плитке
vector<float> avg_grad(const vector<vector<vector<int>>> &cur_tile, int side){
    /*
    parmas: side: 0 - LEFT
                  1 - TOP
                  2 - RIGHT
                  3 - BOTTOM    
            cur_tile - обрабатываемая плитка
    */        

    int height, width; 
    int first_p, second_p;
    vector<vector<int>> G; // array of gradient
    vector<float> mu; // the mean distributions of gradients
    int tmp_csum; // сумма по каналам

    height = cur_tile.size();
    width = cur_tile[0].size();
    
    if (side == 0 || side == 2) {
        G.resize(height, vector<int>(CHANNEL_NUM));
    } else {
        G.resize(width, vector<int>(CHANNEL_NUM));
    }

    if (side == 0){
        for (int i = 0; i < height; i++){
            // cout << cur_tile[i][0][0] << ' ' << cur_tile[i][0][1] << ' ' << cur_tile[i][0][2] << "  ";
            // cout << cur_tile[i][1][0] << ' ' << cur_tile[i][1][1] << ' ' << cur_tile[i][1][2] << "  ";
            // cout << endl;
            for (int j = 0; j < CHANNEL_NUM; j++){
                G[i][j] = cur_tile[i][0][j] - cur_tile[i][1][j];
            }
        }
    } else if (side == 1){
        for (int i = 0; i < width; i++){
            for (int j = 0; j < CHANNEL_NUM; j++){
                G[i][j] = cur_tile[0][i][j] - cur_tile[1][i][j];
            }
        }
    } else if (side == 2){
        for (int i = 0; i < height; i++){
            
            for (int j = 0; j < CHANNEL_NUM; j++){
                
                G[i][j] = cur_tile[i][width-1][j] - cur_tile[i][width-2][j];
                
            }
        }
    } else if (side == 3){
        for (int i = 0; i < width; i++){
            for (int j = 0; j < CHANNEL_NUM; j++){
                G[i][j] = cur_tile[height-1][i][j] - cur_tile[height-2][i][j];
            }
        }
    }

    mu.resize(CHANNEL_NUM);
    for (int i = 0; i < 3; i++){
        tmp_csum = 0;
        for (int j = 0; j < G.size(); j++){
            tmp_csum += G[j][i];
        }
        mu[i] = float(tmp_csum) / float(G.size());
    }

    return mu;
}

vector<vector<int>> difference(const vector<vector<vector<int>>> &cur_tile1, \
                               const vector<vector<vector<int>>> &cur_tile2, int side){
    /*
        side: 0: RIGHT to LEFT
              1: BOTTOM to TOP
              2: LEFT to RIGHT
              3: TOP to BOTTOM
    */
    int height, width; 
    int first_p, second_p;
    vector<vector<int>> GLR; // array of gradient

    height = cur_tile1.size();
    width = cur_tile1[0].size();
    
    if (side == 0) {
        GLR.resize(height, vector<int>(CHANNEL_NUM));
        for (int i = 0; i < height; i++){
            for (int j = 0; j < CHANNEL_NUM; j++){
                GLR[i][j] = cur_tile2[i][0][j] - cur_tile1[i][width-1][j];
            }
        }
    } else if (side == 1) {
        GLR.resize(width, vector<int>(CHANNEL_NUM));
        for (int i = 0; i < width; i++){
            for (int j = 0; j < CHANNEL_NUM; j++){
                GLR[i][j] = cur_tile2[0][i][j] - cur_tile1[height-1][i][j];
            }
        }
    } else if (side == 2) {
        GLR.resize(height, vector<int>(CHANNEL_NUM));
        for (int i = 0; i < height; i++){
            for (int j = 0; j < CHANNEL_NUM; j++){
                GLR[i][j] = cur_tile1[i][width-1][j] - cur_tile2[0][i][j];
            }
        }
    } else if (side == 3) {
        GLR.resize(width, vector<int>(CHANNEL_NUM));
        for (int i = 0; i < width; i++){
            for (int j = 0; j < CHANNEL_NUM; j++){
                GLR[i][j] = cur_tile1[height-1][i][j] - cur_tile2[0][i][j];
            }
        }
    }
    return GLR;

}


float covariance(const vector<float> &v1, const vector<float> &v2){
    float sum = 0;
    float mean1, mean2;

    mean1 = accumulate(v1.begin(), v1.end(), 0.0) / v1.size();
    mean2 = accumulate(v2.begin(), v2.end(), 0.0) / v2.size();

    for(int i = 0; i < v1.size(); i++)
        sum += (v1[i] - mean1) * (v2[i] - mean2);

    return sum / (v1.size() - 1);
}

vector<vector<float>> dot_prod(const vector<vector<float>> m1, const vector<vector<float>> m2){
    vector<vector<float>> prod;
    prod.resize(m1.size(), vector<float>(m2[0].size()));
 
    // Loop for calculate dot product
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            prod[i][j] = 0;
 
            for (int k = 0; k < m2.size(); k++) {
                prod[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    return prod;

}


// compatibility measure
vector<vector<float>> comp_m(const vector<vector<int>> &differ, const vector<float> &avg_grad){
    vector<vector<float>> tmp_vec, tmp_vec_tr; // tmp_vec - разность между differ и avg_grad; tmp_vec_tr - транспонированный вектор
    
    tmp_vec.resize(differ.size(), vector<float>(CHANNEL_NUM));
    for(int i = 0; i < differ.size(); i++){
        for(int j = 0; j < CHANNEL_NUM; j++){
            tmp_vec[i][j] = differ[i][j] - avg_grad[j];
        }
    }

    // транспонируем 
    tmp_vec_tr.resize(CHANNEL_NUM, vector<float>(differ.size()));
    for (int i = 0; i < tmp_vec.size(); i++){
        for (int j = 0; j < tmp_vec[i].size(); j++){
            tmp_vec_tr[j][i] = tmp_vec[i][j];
        }
    }

    // covariance matrix
    vector<vector<float>> cov, inverse_cov; 
    cov.resize(CHANNEL_NUM, vector<float>(CHANNEL_NUM));
    for(int i = 0; i < cov.size(); i++){
        for(int j = 0; j < cov[i].size(); j++){
            cov[i][j] = covariance(tmp_vec_tr[i], tmp_vec_tr[j]);
            // cout << cov[i][j] << ' ';
        }
        // cout << endl;
    }

    // inverse matrix

    float determinant;
    inverse_cov.resize(CHANNEL_NUM, vector<float>(CHANNEL_NUM));
    //finding determinant
    
    for(int i = 0; i < CHANNEL_NUM; i++) {
        determinant = determinant + (cov[0][i] * (cov[1][(i+1)%3] * cov[2][(i+2)%3] - cov[1][(i+2)%3] * cov[2][(i+1)%3]));
    }

    for(int i = 0; i < CHANNEL_NUM; i++){
        for(int j = 0; j < CHANNEL_NUM; j++){
            inverse_cov[i][j] = ((cov[(j+1)%3][(i+1)%3] * cov[(j+2)%3][(i+2)%3]) - (cov[(j+1)%3][(i+2)%3] * cov[(j+2)%3][(i+1)%3]))/ determinant;
            // cout << inverse_cov[i][j] << ' ';
        }
        // cout << endl;
    }

    // dot product
    vector<vector<float>> product;

    tmp_vec = dot_prod(tmp_vec, inverse_cov);
    product = dot_prod(tmp_vec, tmp_vec_tr);

    return product;
}


// float SSD(const vector<vector<vector<int>>> &cur_tile1, const vector<vector<vector<int>>> &cur_tile2, int side){
//     /**
//      * side: 2 - LEFT/RIGHT; RIGHT/LEFT 
//      *       3 - TOP/BOTTOM; BOTTOM/TOP
//      * 
//      */

//     vector<vector<int>> diff;
//     vector<int> q_sum; 
//     diff = difference(cur_tile1, cur_tile2, side);
//     for (int i = 0; i < diff.size(); i++){
//         for (int j = 0; j < diff[0].size(); j++){
//             diff[i][j] = pow(diff[i][j], 2);
//         }
//     }

//     return ssd;
// }

int main(int argc, char* argv[]){
    vector<vector<vector<int>>> image;
    string tile_dir = argv[1]; // tiles directory
    vector<vector<vector<vector<int>>>> tiles; 
    int tile_ind = 0; 
    // cout << tile_dir << endl;
   
    // read tiles
    while (true){
        string str_tile_ind = std::to_string(tile_ind);
        int lead_zeros_num = 4 - str_tile_ind.length(); // zeros ahead
        str_tile_ind.insert(0, lead_zeros_num, '0');
        str_tile_ind += IMG_EXTENSION;
        string image_path = tile_dir + "/" + str_tile_ind;
        ifstream file(image_path.c_str());
        if (!file.is_open())
            break;
        vector<vector<vector<int>>> cur_tile;

        readImage(file, cur_tile);

        tiles.push_back(cur_tile);
        tile_ind += 1;
    }

    // cout << tiles.size() << endl; // количество считанных плиток
    // расчет Mahalanobis Gradient Compatibility (MGC) 

    vector<vector<vector<int>>> cur_tile_first, cur_tile_second;
    vector<vector<int>> grad1, grad2;
    vector<float> mu_first, mu_second, tmp_vec; 
    vector<vector<float>> comp_measure1, comp_measure2, comp_measure; // compatibility measure
    vector<float> similarity, similarity1, tmp_similarity;
    vector<vector<float>> mgc_all; // mgc для всех конфигураций
    int rotation_pos; // в какой позиции наименьший показатель similarity, крутим по часовой
    int index_sim_tile; // индекс наиболее подходящей детали 
    int position, param1, param2; // отношение между плитками: 0 - LEFT/RIGHT, 1 - BOTTOM/TOP
    float mgc; // мера
    float sim_num, tmp_sim_num;

    mgc_all.resize(0, vector<float>(5));
    for(int i = 0; i < tiles.size(); i++){
        cur_tile_first = tiles[i];
        for(int j = i + 1; j < tiles.size(); j++){
            cur_tile_second = tiles[j];
            position = 0;
            

        }
    }



    cur_tile_first = tiles[0];
    cur_tile_second = tiles[1];

    position = 0; // задача позиции

    if (position == 0){
        param1 = 2;
        param2 = 0;
    } else {
        param1 = 3;
        param2 = 1;
    }

    mu_first = avg_grad(cur_tile_first, param1);
    mu_second = avg_grad(cur_tile_second, param2);

    grad1 = difference(cur_tile_first, cur_tile_second, param2);
    grad2 = difference(cur_tile_first, cur_tile_second, param1);

    comp_measure1 = comp_m(grad1, mu_first);
    comp_measure2 = comp_m(grad2, mu_second); 

    comp_measure.resize(comp_measure1.size(), vector<float>(comp_measure1[0].size()));

    mgc = 0;
    for(int i = 0; i < comp_measure1.size(); i++){
        for(int j = 0; j < comp_measure1[0].size(); j++){
            comp_measure[i][j] = comp_measure1[i][j] + comp_measure2[i][j];
            mgc += comp_measure[i][j];
        }
    }

    cout << "mgc: " << mgc << endl;

    

    

    

    // cout << "product: " << endl;
    // for (int i = 0; i < comp_measure.size(); i++){
    //     for (int j = 0; j < comp_measure[0].size(); j++){
    //         cout << comp_measure[i][j] << ' ';
    //     }
    //     cout << endl;
    // }

    // cout << "mu1_sum:" << endl;
    // for (int i = 0; i < 3; i++){
    //     cout << mu_first[i] << ' ';
    // }
    // cout << endl;

    // cout << "diff" << endl;
    // for (int i = 0; i < grad1.size(); i++){
    //     for (int j = 0; j < CHANNEL_NUM; j++){
    //         cout << grad1[i][j] << ' ';
    //     }
    //     cout << "  ";
    //     for (int j = 0; j < CHANNEL_NUM; j++){
    //         cout << grad2[i][j] << ' ';
    //     }
    //     cout << endl;
    // }

    // cout << "mu1_sum:" << endl;
    // for (int i = 0; i < 3; i++){
    //     cout << mu_first[i] << ' ';
    // }
    
    // cout << endl << "mu2_sum:" << endl;
    // for (int i = 0; i < 3; i++){
    //     cout << mu_second[i] << ' ';
    // }
    

// find minimal h and w
//     int min_h = tiles[0].size();
//     int min_w = tiles[0][0].size();

//     cout << "min_h: " << min_h << endl;
//     cout << "min_w: " << min_w << endl;
//     cout << "third dim: " << tiles[0][0][0].size() << endl;

//     for (int i = 1; i < tiles.size(); i++){
//         int cur_h = tiles[i].size();
//         min_h = min(min_h, cur_h);
//         int cur_w = tiles[i][0].size();
//         min_w = min(min_w, cur_w);
//     }

//     cout << "________" << endl;
//     cout << "min_h: " << min_h << endl;
//     cout << "min_w: " << min_w << endl;

// // generate nodes
//     vector<vector<vector<int>>> resultImage(H, vector<vector<int>>(W, vector<int>(CHANNEL_NUM)));
//     vector<pair<int, int>> nodes;
//     for (int x = 0; x < W; x+= min_w){
//         for (int y = 0; y < H; y += min_h){
//             nodes.push_back({x, y});
//         }
//     }

//     cout << "nodes_size: " << nodes.size() << endl;
//     cout << "tiles size: " << tiles.size() << endl;
// // fill image with tiles
//     for (int i = 0; i < nodes.size(); i++)    {
//         int x_left_top = nodes[i].first;
//         int y_left_top = nodes[i].second;
//         if (i == tiles.size())
//             break;
//         for (int x = 0; x < min_w; x++){
//             for (int y = 0; y < min_h; y++){
//                 resultImage[y_left_top + y][x_left_top + x] = tiles[i][y][x];
//             }
//         }
//     }

//     string output_path = "image.ppm";
//     writeImage(resultImage, output_path);
}
