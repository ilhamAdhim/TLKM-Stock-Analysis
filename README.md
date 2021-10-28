# Laporan Proyek Machine Learning <br> Muhammad Ilham Adhim

## Project Overview

Seiring dengan perkembangan internet dan adanya kasus pandemi COVID-19, semakin banyak orang yang menyadari pentingnya manajemen finansial dan memulai investasi. Namun dengan banyaknya perusahaan yang ada, pemula akan kebingungan untuk memulai investasi saham di perusahaan yang cocok. Jika kita investasi ke perusahaan tanpa mengenali profil perusahaan dan track record manajemen keuangan dari perusahaan tersebut, besar kemungkinan kita akan merugi dari segi waktu, effort, dan biaya. Oleh karena itu, sebagai pemula direkomendasikan untuk mencoba saham *Bluechip*

Untuk [Daftar saham Bluechip](https://superyou.co.id/blog/keuangan/rekomendasi-saham-blue-chip/) di Indonesia cukup bervariasi. Dalam pengerjaan submission ini, saya mencoba untuk memprediksi harga saham TLKM oleh PT. Telkom Indonesia (Persero) Tbk. menggunakan data yang ada.

## Business Understanding
### Problem Statements
1. Mengamati pergerakan saham *bluechip* dari PT. Telkom Indonesia (Persero) Tbk.
2. Memprediksi pergerakan saham PT. Telkom Indonesia (Persero) Tbk. menggunakan data uji yang ada

### Goals
Membuat prediksi harga saham PT Telkom Indonesia (Persero) Tbk. berdasarkan dataset yang ada.

### Solution statements
Karena dataset terkait hanya berisi tentang data tanggal dan harga, maka solusi yang tepat untuk masalah ini adalah dengan menggunakan pendekatan Time Series.

Saya menggunakan 1 buah layer LSTM (Long Short Term Memory) dalam model.LSTM adalah jenis jaringan saraf berulang yang memiliki kemampuan untuk mengingat atau melupakan output dari data yang melalui arsitekturnya. Ini dilakukan tanpa mengubah konteks dari data yang ada. Dengan pendekatan ini, LSTM mampu mengatasi masalah RNN, yang mana RNN tidak mampu memprediksi kata yang disimpan dalam memori jangka panjang. Dengan bertambahnya panjang celah, RNN tidak memberikan kinerja yang efisien. Berbeda dengan LSTM yang dapat secara default menyimpan informasi. Dengan kinerja seperti ini, LSTM cocok untuk digunakan dalam proses analisa dan prediksi data deret waktu. Beberapa keuntungan untuk menggunakan LSTM untuk kasus Time Series adalah:

1. Tidak ada prasyarat tertentu dalam implementasi model
2. Dapat bekerja dengan baik untuk neural network dengan fungsi non-linear
3. Cocok untuk digunakan di dataset yang banyak
4. Dapat mengatur parameter tuning secara kustom agar menyesuaikan bentuk data.

Menurut [referensi](https://www.springml.com/blog/time-series-forecasting-arima-vs-lstm/) yang saya baca, Saya memutuskan untuk menggunakan pembuatan model menggunakan layer **LSTM (Long Short Term Memory).** Untuk penggunaan [optimizer yang cocok](https://deepdatascience.wordpress.com/2016/11/18/which-lstm-optimizer-to-use/) di case Time Series, Saya akan menggunakan **Adam Optimizer** karena secara keseluruhan, optimizer ini sangat bagus jika dibandingkan dengan optimizer lain.


## Data Understanding
Untuk submission ini, saya mengambil data dari [Kaggle](https://www.kaggle.com) yang bernama **[Indonesian Government Owned Company Stock Price](https://www.kaggle.com/fawwazzainiahmad/indonesian-government-owned-company-stock-price?select=TLKM.JK.csv)**. Berikut adalah daftar kolom di file CSV yang tersedia:

  * Date - Tanggal trading saham TLKM (datatype : string object)
  * Open - Harga ketika pertama kali diumumkan di tanggal tersebut (datatype : float64)
  * High - Harga tertinggi di tanggal tersebut (datatype : float64)
  * Low -  Harga terendah di tanggal tersebut (datatype : float64)
  * Close - Harga saham ketika diakhir period (datatype : float64)
  * Adj Close - Close value setelah mempertimbangkan dividen dan stock split (datatype : float64)
  * Volume - Jumlah transaksi saham di tanggal tersebut (datatype : float64)
  
<br>

![Image data Overview](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/data_overview.png?raw=true)


Dari data tersebut, terlihat bahwa rata-rata harga saham TLKM disajikan sangat lengkap mulai dari Harga Open sampai Adj Close nya periode 2004 sampai 2020. Disertai dengan informasi penting lainnya, seperti harga saham tertinggi dan terendah dalam durasi tersebut.

![Grafik Saham TLKM 2004 - 2021](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/data_understanding.png?raw=true)

Dari grafik tersebut, dapat diambil kesimpulan bahwa harga saham TLKM mengalami perubahan secara signifikan dalam durasi tersebut. Pada tahun 2004 sampai 2008 merupakan lonjakan harga saham TLKM pertama, kemudian terjadi peningkatan yang sangat drastis di tahun 2012 - Q1 2018. Kemudian mengalami penurunan yang cukup signifikan di akhir tahun 2019 sampai 2 Oktober 2020 karena pengaruh COVID-19.

## Data Preparation
Dalam tahap ini, saya menyiapkan dataframe yang telah menyimpan data dari CSV tersebut untuk dilakukan beberapa pengecekan, pertama kita perlu memeriksa adanya null values. Ini perlu dilakukan untuk menjaga akurasi dari prediksi model yang akan kita lakukan di proses pelatihan data. 

Berikut hasil cek data null oleh library **pandas** : <br>

![Check null values](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/ada-null-values.png?raw=true)

Dari 3980 data, terdapat 3944 data yang tidak ada null valuesnya, ini artinya ada beberapa data yang null. Untuk mengatasinya, kita bisa menghapus row yang null dengan **dropna()** dari library **pandas**


```
df_new = df.dropna(how='any',axis=0) 
df_new
```


Kemudian, kita cek juga untuk duplikasi data. Berikut hasil cek duplikasi data oleh library **pandas** : <br>

![Check duplicate values](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/check-duplicate-values.png?raw=true)

Selain pengecekan data, kita juga perlu untuk mengatur skala data. Hal ini perlu dilakukan agar skor MAE kita tidak menjadi terlalu besar, jika hal ini terjadi, akan mengakibatkan prediksi kita sangat buruk. Oleh karena itu, saya melakukan skala data menggunakan MinMax Scaler. Berikut formula dari MinMax Scaler: <br>

![MinMax Scaler Formula](https://i.stack.imgur.com/ruy6L.png)

Sebagai rangkuman, langkah yang telah saya lakukan untuk tahap ini adalah: 

1. Penghapusan missing values 
2. Penghapusan duplikat data
3. Penskalaan Data dengan MinMax Scaler
4. Train Test Split dataframe dengan ratio 80% data latih dan 20% data uji.

## Modeling

Dari banyaknya opsi penggunaan model yang ada untuk kasus Time Series, saya mencoba mengimplementasikan LSTM dalam pembuatan model. 

Dalam prosesnya, kita telah mengetahui bahwa LSTM ini merupakan perbaikan dari RNN Tradisional dimana LSTM mampu menyimpan nilai yang penting dan menghapus nilai yang tidak penting dalam jangka waktu yang lama secara default.

Semakin kompleks sebuah model ML, maka kemungkinan model tersebut mengalami overfitting pun semakin tinggi. Walaupun secara arsitektur sudah cocok dengan data, menggunakan loss function yang tepat, dan metrik yang sesuai, masih ada kemungkinan overfitting. Oleh karena itu, selain LSTM saya juga menggunakan Dropout layer untuk mencegah terjadinya overfitting selama proses pelatihan data. Simpelnya, dropout layer yang berperan sebagai perantara hidden layer dan output layer ini dimatikan secara bergantian selama proses pelatihan data berlangsung. Dalam project ini, menggunakan dropout value sebesar 0.5, berikut ilustrasinya:
<br>
<br>
![image](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20200803125202b077a1253a77def9b9e4ae6b553bc1cc.gif)

Secara keseluruhan, alur dari arsitektur model ini adalah LSTM layer sebagai input layer, kemudian melewati dropout layer dengan value sebesar 0.5 untuk meningkatkan variasi output, barulah menggunakan Dense Layer dengan 1 unit perceptron sebagai output layernya.

## Model Evaluation

Berikut visualisasi untuk nilai MAE dan loss value di tahap pelatihan dan pengujian <br><br>
![Model Evaluation Result](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/model_evaluation.png?raw=true)

Berikut visualisasi untuk prediksi data latih harga saham TLKM dibandingkan dengan data aslinya dalam periode 28 September 2004 - 29 November 2016 (80% dataset) <br><br>
![Prediction Result](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/model_prediction.png?raw=true)


## Evaluation

- ***Mean Absolute Error*** <br><br>
![MAE Formula](https://github.com/ilhamadhim/TLKM-Stock-Analysis/blob/master/assets/MAE_Formula.png?raw=true)

Metrik ini digunakan untuk mengetahui kesalahan model atau memberitahu seberapa besar error model yang sudah di latih kepada data yang akan diuji.

- ***Mean Squared Error***:  <br><br>
![MSE Formula](https://miro.medium.com/max/808/1*-e1QGatrODWpJkEwqP4Jyg.png)<br>
Fungsi loss yang paling sederhana dan sering digunakan untuk kasus regresi

<br>

Model deep learning yang telah dibuat dapat melakukan proses training data dengan metrik dan loss function tersebut. Dalam prosesnya, terlihat hasil MAE yang relatif kecil yaitu sekitar 0.0290. Hal ini menunjukan bahwa model ini memiliki error dibawah 1.8%

## Penutup
Demikian laporan dan metrik dari implementasi Machine Learning untuk analisis harga saham TLKM oleh PT. Telkom Indonesia Tbk. Terimakasih telah membaca laporan ini, semoga bermanfaat.