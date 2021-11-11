# Laporan Proyek Machine Learning - Farischa Makay

## Domain Proyek 

![Diabetes](https://www.tagar.id/Asset/uploads/486854-gambar-diabetes.jpeg)

 Diabetes adalah penyakit kronis yang ditandai dengan ciri-ciri berupa tingginya kadar gula (glukosa) darah. Glukosa merupakan sumber energi utama bagi sel tubuh manusia. Glukosa yang menumpuk di dalam darah akibat tidak diserap sel tubuh dengan baik dapat menimbulkan berbagai gangguan organ tubuh. Jika diabetes tidak dikontrol dengan baik, dapat timbul berbagai komplikasi yang membahayakan nyawa penderita.  
 Tujuan dari analisis ini adalah untuk membangun model Machine Learning (K-NN) untuk memprediksi secara akurat apakah pasien dalam dataset menderita diabetes atau tidak dengan menggunakan variabel-variabel independen yang mempengaruhi terhadap keputusan pasien tersebut diagnosis menderita diabetes atau tidak.


## Business Understanding

### Problem Statements
- Bagaimana cara membuat model machine learning untuk mengklasifikasikan data pasien mengalami penyakit diabetes?
- Bagaimana cara memprediksi pasien dinyatakan positif ataupun negatif terhadap diabetes?

### Goals
- Memprediksi penderita penyakit diabetes dalam dataset dengan tingkat akurasi lebih dari 80%.
- Mengidentifikasi pasien positif ataupun negatif terhadap diabetes.

### Solution statements
Solusi yang diterapkan untuk mencapai tujuan dari proyek ini, diantaranya :
1.  Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
 - Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
 - Melakukan standardisasi data pada semua fitur data.
2. Proses pemodelan. Membuat model dengan algoritma **K-Nearest Neighbor**.
K-Nearest Neighbor adalah algoritma supervised learning dimana hasil dari instance yang baru diklasifikasikan berdasarkan mayoritas dari kategori k-tetangga terdekat. Tujuan dari algoritma ini adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data. Algoritma k-Nearest Neighbor menggunakan Neighborhood Classification sebagai nilai prediksi dari nilai instance yang baru. 
Kelebihan yang dimiliki KNN antara lain:
- K-NN cukup intuitif dan sederhana: Algoritma K-NN sangat sederhana untuk dipahami dan juga mudah diimplementasikan. Akurasi tinggi (relatif).
- K-NN tidak memiliki asumsi: K-NN adalah algoritma non-parametrik yang berarti ada asumsi yang harus dipenuhi untuk mengimplementasikan K-NN. Model parametrik seperti regresi linier memiliki banyak asumsi yang harus dipenuhi oleh data sebelum dapat diimplementasikan yang tidak terjadi pada K-NN.
- Dapat digunakan baik untuk Klasifikasi dan Regresi: Salah satu keuntungan terbesar K-NN adalah bahwa K-NN dapat digunakan baik untuk masalah klasifikasi dan regresi. Karena kasus pada data yang digunakan termasuk kasus hubungan pengaruh variabel prediktor terhadap variabel hasil, maka data tersebut cocok menggunakan metode K-NN.
Kekurangan yang dimiliki KNN antara lain:
- Algoritme menjadi lebih lambat secara signifikan karena jumlah contoh dan/atau prediktor/variabel yang meningkat. 

3. Membandingkan Error Rate dengan K-value untuk menemukan nilai optimal dari K

## Data Understanding
![Dataset](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/dataset.png)
 [Dataset](https://www.kaggle.com/mathchi/diabetes-data-set) yang saya gunakan berasal dari [Kaggle](https://www.kaggle.com/). Dengan informasi sebagai berikut:
 - Jenis dan ukuran berkas : CSV(24 kB)
 -  Memiliki 9 kolom : Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, dan Outcome.
 -  Memiliki 768 row data.
   
Visualisasi jumlah data pasien yang mengidap penyakit diabetes :
![Positive/Negative](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/pvn.png)
Dari visualisasi diatas, kita dapat mengetahui bahwa pasien yang terkena penyakit diabetes ada 268 pasien.
Variabel-variabel pada Dataset dataset adalah sebagai berikut:
- Pregnancies : Perhitungan berapa kali hamil pada pasien wanita.
- Glukosa : Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral. Hasil Normal untuk Diabetes -> Kadar glukosa dua jam kurang dari 140 mg/dL, Hasil Gangguan untuk Diabetes -> Kadar glukosa dua jam 140 hingga 200 mg/dL, Hasil Abnormal (Diagnostik) untuk Diabetes -> Kadar glukosa dua jam lebih besar dari 200 mg/dL.
- BloodPressure : Tekanan darah diastolik (mm Hg). Tekanan darah diastolik normal lebih rendah dari 80. Angka 90 atau lebih tinggi berarti  memiliki tekanan darah tinggi. Normal: Sistolik di bawah 120 dan diastolik di bawah 80. Peningkatan: Sistolik 120–129 dan diastolik di bawah 80. Hipertensi stadium 1: Sistolik 130-139 dan diastolik 80-89. Hipertensi stadium 2: Sistolik 140-plus dan diastolik 90 atau lebih. Krisis hipertensi: Sistolik lebih tinggi dari 180 dan diastolik di atas 120.
- SkinThickness : Ketebalan lipatan kulit trisep (mm). Untuk normalnya, pada Pria 12,5 mm dan 
Wanita 16,5 mm
- Insulin : insulin serum 2 Jam (mu U/ml). Insulin dihasilkan oleh pankreas. Setelah memasuki sel, zat gula ini akan dibakar menjadi energi yang bisa Anda pakai. Untuk normalnya, sebelum tidur atau 2 jam setelah makan: kurang dari 140 mg/dL.
- BMI : Body Mass Index adalah perhitungan sederhana menggunakan tinggi dan berat badan seseorang. Rumusnya adalah BMI = kg/m2 dimana kg adalah berat badan seseorang dalam kilogram dan m2 tingginya dalam meter persegi.
- DiabetesPedigreeFunction : Fungsi silsilah diabetes. Ini memberikan informasi tentang riwayat diabetes pada kerabat dan hubungan genetik kerabat tersebut dengan pasien. Fungsi Silsilah yang lebih tinggi berarti pasien lebih mungkin untuk menderita diabetes. 
- Age : Usia (tahun).
- Outcome :  Menentukan pasien mangidap penyakit diabetes (nilai 1) atau tidak  (nilai 0). 
## Data Preparation
Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :
- Mengatasi nilai bernilai nol pada beberapa kolom dengan median sesuai target dan melakukan handling outlier. 
 Untuk mengatasinya, saya terlebih dahulu mengganti nilai nol pada data Glukosa, Tekanan Darah, Ketebalan Kulit, Insulin, BMI dengan nilai NaN. Setelah itu saya menngganti nilai NaN dengan nilai mean/rata-rata. Data rata-rata kolom dipilih karena merupakan data yang dipastikan bukan data pencilan. Sehingga dengan menganggap data kosong sebagai data rata-rata, model tetap dapat memperoleh informasi dari data yang ada pada kolom lainnya. Semua proses tersebut dilakukan dengan slicing data dengan kondisi menggunakan pandas. Kemudian, untuk handling outlier dapat menggunakan metode IQR untuk mengidentifikasi outlier untuk mengatur "pagar" di luar Q1 dan Q3. Setiap nilai yang berada di luar pagar ini dianggap outlier. Untuk membangun pagar ini kita mengambil 1,5 kali IQR dan kemudian mengurangi nilai ini dari Q1 dan menambahkan nilai ini ke Q3. Ini memberi kita pos pagar minimum dan maksimum yang kita bandingkan dengan setiap pengamatan. Setiap pengamatan yang lebih dari 1,5 IQR di bawah Q1 atau lebih dari 1,5 IQR di atas Q3 dianggap outlier. Ini adalah metode yang digunakan untuk mengidentifikasi outlier secara default.

- Mengatasi data yang tidak seimbang jumlahnya dengan label lain menggunakan teknik resample
Dataset yang tidak seimbang pada data kategori akan menyebabkan model yang dibuat menjadi bias terhadap suatu kategori yang memiliki data lebih banyak. Oleh karena itu diperlukan teknik manipulasi data, dan yang digunakan di sini adalah teknik resample. Prosesnya adalah dengan memasukan kolom yang memiliki data paling sedikit pada fungsi resample, kemudian fungsi resample akan menghasilkan data baru dari data yang sudah ada sebelumnya sampai jumlah datanya sama dengan data mayoritas dari label selainnya. Setelah itu selesai, masukan datanya kedalam dataset agar menjadi satu kesatuan data.
 
- Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
Agar dapat menguji performa model pada data sebenarnya, maka perlu dilakukan pembagian dataset kedalam dua atau tiga bagian. Pada proyek ini dilakukan dua bagian saja yakni pada data latih dan data uji dengan rasio 80:20. Data latih dilakukan sepenuhnya untuk melatih model, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. Pada bagian ini dipastikan juga pembagian label kategorikal haruslah sama banyak pada data latih dan data uji. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn.
- Melakukan standardisasi data pada semua fitur data.
Tahap terakhir dengan melakukan standarisasi data. Jika salah satu variabel memiliki jarak yang dengan rentang yang jauh(skala berbeda),maka nilai tersebut harus disesuaikan. Jadi jarak antara semua variabel harus distandarisasi agar jarak akhir yang didapatkan proposional. Hal ini akan membuat semua fitur numerik berada dalam skala data yang sama juga membuat komputasi dari pembuatan model dapat berjalan lebih cepat karena rentang datanya hanya antara 0-1. Untuk melakukan standarisasi data, digunakan fungsi StandardScaler. 

## Modeling
Pada tahap ini saya membuat model dengan menggunakan modul scikit-learn yakni KNeighborsClassifier tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya.
Hasil prediksi yang mengindikasikan pasien mengalami sakit diabetes dengan menggunakan KNN, sebagai berikut :
![Result1](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/sebelum.jpg)
Dengan matrix dan tingkat akurasi sebagai berikut :
![Result1](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/pred1.jpg)

Dapat dilihat tingkat akurasi yang dihasilkan sudah baik, yakni sebesar 84%, Namun kita dapat meningkatkan lagi tingkat akurasi. Membandingkan Error Rate dengan K value, menghasilkan grafik sebagai berikut :
![Result2](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/grafik.jpg)
Dengan memperthatikan gambar plot diatas, dapat diketahui bahwa Mean Error terkecil yaitu ketika nilai K value sama dengan 19. Dengan menggunakan k = 19, prediksi yang dihasilkan sebagai berkut :
![Result3](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/sesudah.jpg)
Dengan matrix dan tingkat akurasi sebagai berikut :
![Result4](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/pred2.jpg)

Dapat kita lihat setelah menggunakan k=19, tingkat akurasi naik menjadi 90%. Visualisasi Confussion matrix sebagai berikut :
- Model KNN sebelum ditingkatkan akurasinya
![Result5](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/cm1.jpg)
- Model KNN setelah ditingkatkan akurasinya
![Result6](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/cm2.jpg)

Dengan hasil diatas, maka model yang ditingkatkan akurasinya merupakan model yang dipilih untuk digunakan.

## Evaluation
Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi, f1-score, recall dan precision.
![Result7](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/eval.jpg)
- Akurasi
Akurasi merupakan metrik untuk menghitung nilai ketepatan model dalam memprediksi data dengan data yang sebenarnya. Pada proyek, setelah menerapkan peningkatan akurasi, akurasi yang di peroleh hasil sebesar 90%. Tingkat akurasi yang dihasilkan pada data sudah termasuk dalam kategori sangat baik. Kelebihan dari metriks ini adalah sering digunakan dalam kasus pembuatan model klasifikasi baik itu klasifikasi dua kelas, atau kategori. Kekurangan dari metrik ini adalah dapat bersifat 'menyesatkan' pada data yang tidak seimbang.
- f1-score
Precision merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua prediksi model berlabel positif. Pada proyek hasil f1-score yang dihasilkan pada data ialah 89%. Kelebihan dari metriks ini berfokus pada bagaimana performa (prediksi) model terhadap label data positif, kekurangannya metriks ini tidak memperhitungkan label negatifnya.
- recall 
Recall merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua label data positif. Pada proyek hasil recall yang dihasilkan pada data ialah 85%. Kelebihan dari metriks ini menghitung bagian negatif dari prediksi label positif (tidak seperti precision). Tetapi kekurangannya ketika semua prediksi = 1 maka recall akan bernilai 1 (tidak memperhitungkan prediksi negatif).
- precision
Precision merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua prediksi model berlabel positif. Pada proyek hasil recall yang dihasilkan pada data ialah 93%. Lalu bagaimana cara menghitungnya, pertama-tama kita perlu mengenali dulu istilah TP,TN,FP,FN.
--	True Positive (TP): Jumlah prediksi positif yang benar terhadap jumlah positif yang sebenarnya.
--	False Positive (FP): Jumlah prediksi positif yang salah.
--	True Negative (TN): Jumlah prediksi negatif yang benar terhadap jumlah negatif yang sebenarnya.
--	False Negative (FN): Jumlah prediksi negatif yang salah.
Kelebihan dari metriks ini berfokus pada bagaimana performa (prediksi) model terhadap label data positif, kekurangannya metriks ini tidak memperhitungkan label negatifnya.

Perhitungan metriks diatas dengan rumus sebagai berikut :
![Result7](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/calculate.png)
![Result8](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/f-measure.jpg)


# *Referensi*
- Marianti, (2020, Maret 27). Diabetes. Alodokter. *https://www.alodokter.com/diabetes*
- Harrison, O. (2019, July 14). Machine Learning Basics with the K-Nearest Neighbors Algorithm. Medium. *https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761*
-  Frost, J. (2021, April 5). Guidelines for Removing and Handling Outliers in Data. Statistics By Jim. *https://statisticsbyjim.com/basics/remove-outliers/*

**---Ini adalah bagian akhir laporan---**

_Catatan:_
_Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
