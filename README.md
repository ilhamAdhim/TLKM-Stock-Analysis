# Laporan Proyek Machine Learning <br> Muhammad Ilham Adhim

## Project Overview

Seiring dengan perkembangan internet dan adanya kasus pandemi COVID-19, semakin banyak orang yang menyadari pentingnya manajemen finansial dan  memulai investasi ketika memiliki *cashflow* yang stabil dan mempunyai dana darurat. Namun dengan banyaknya perusahaan yang ada, sebagai pemula akan kebingungan untuk memulai investasi saham di perusahaan yang cocok. Jika kita investasi ke perusahaan tanpa mengenali profil perusahaan dan track record manajemen keuangan dari perusahaan tersebut, besar kemungkinan kita akan merugi dari segi waktu, effort, dan biaya. Oleh karena itu, sebagai pemula direkomendasikan untuk mencoba reksadana dan saham *Bluechip*

Dikutip dari artikel [Kompas](https://money.kompas.com/read/2021/08/24/161914926/mengenal-apa-itu-saham-blue-chip-dan-contohnya?page=all)

> "Saham blue chip adalah jenis saham dari perusahaan dengan kondisi keuangan prima, serta beroperasi selama bertahun lamanya. Kondisi keuangan prima terukur dari pendapatan perusahaan yang tumbuh stabil setiap tahun, dan kerap membagikan dividen kepada investor."

Untuk [Daftar saham Bluechip](https://superyou.co.id/blog/keuangan/rekomendasi-saham-blue-chip/) di Indonesia cukup bervariasi. Dalam pengerjaan submission ini, saya mencoba untuk memprediksi harga saham TLKM oleh PT. Telkom Indonesia  (Persero) Tbk. menggunakan data yang ada.



## Business Understanding
### Problem Statements
Membuat prediksi harga saham PT Telkom Indonesia (Persero) Tbk. berdasarkan dataset yang ada.

### Goals
Memprediksi harga saham TLKM

### Solution statements
Karena dataset terkait hanya berisi tentang data tanggal dan harga, maka solusi yang sangat tepat untuk masalah ini adalah dengan menggunakan pendekatan Time Series. 

Untuk kasus Time Series sendiri, sebenarnya ada banyak model yang bisa digunakan, seperti [Autoregressive Integrated Moving Average (ARIMA)](https://daps.bps.go.id/file_artikel/77/arima.pdf), [Vector Autoregression (VAR)](https://www.aptech.com/blog/introduction-to-the-fundamentals-of-vector-autoregressive-models/), dan masih banyak lagi. 

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



Dari data di atas terlihat bahwa rata-rata atau mean harga minya dari tahun 1987 hingga 2021 adalah US$46,352962. Untuk harga maximum atau tertingginya dari tahun 1987 hingga 2021 adalah sebesar US$143,95. Dan untuk harga minimum atau terendahnya menyentuh harga US$9,1.