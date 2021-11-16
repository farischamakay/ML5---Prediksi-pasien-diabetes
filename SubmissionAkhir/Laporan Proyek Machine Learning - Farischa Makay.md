# Laporan Proyek Machine Learning - Farischa Makay
## Project Overview
![1](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/appstore.png)
Sistem rekomendasi yang dibuat pada proyek ini ialah sistem rekomendasi aplikasi bagi pengguna Apple App Store. Apple App Store adalah platform distribusi aplikasi untuk iOS yang dikembangkan dan dikelola Apple Inc. Layanan ini memungkinkan pengguna menjelajah dan mengunduh aplikasi yang dikembangkan dengan Apple iOS SDK. Aplikasi dapat diunduh langsung ke sebuah perangkat iOS atau komputer pribadi (Macintosh atau PC) melalui iTunes. 
Munculnya banyak aplikasi membuat pengguna kebingungan untuk memilih aplikasi mana yang perlu diunduh untuk memenuhi kebutuhannya. Sehingga perilaku pengguna dalam mengunduh aplikasi sering mengikuti tren. Hal tersebut termasuk kerugian bagi beberapa pengembang yang membuat aplikasi sudah memiliki fitur yang tidak ada di aplikasi yang sedang digunakan banyak orang namun karena tidak dikenal maka aplikasi tersebut kurang dipakai pengguna. Oleh karena itu diperlukan sebuah sistem rekomendasi agar hal-hal seperti ini tidak terjadi. Selain sebagai fungsi periklanan, sistem rekomendasi juga membuat aplikasi-aplikasi yang jarang diunduh pengguna menjadi kembali dikenal karena sebelumnya sulit untuk ditemukan atau bahkan mempermudah pengguna menemukan aplikasi yang diharapkan.
## Business Understanding
### Problem Statements
- Bagaimana cara membuat sistem rekomendasi aplikasi untuk pengguna di Apple App Store?
- Apa sistem rekomendasi yang dapat digunakan agar memberikan rekomendasi aplikasi yang diharapkan? 
### Goals
- Membuat sistem rekomendasi aplikasi yang baik dengan algoritma yang tepat bagi pengguna di Apple App Store.
- Memberikan rekomendasi untuk aplikasi yang kemungkinan disukai pengguna.


### Solution Approach
- Untuk bagian persiapan data (sebelum dimasukkan ke model) dilakukan beberapa teknik diantaranya :
  - Konversi label kategori menjadi one-hot encoding.
  - Standarisasi label numerik.
Penjelasan lengkap mengenai persiapan data dapat dilihat lebih lengkap pada bagian Data Preparation.
- Menggunakan pendekatan sistem rekomendasi *Content Based Filtering*. Ide dari sistem rekomendasi berbasis konten (content-based filtering) adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu.Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
- Membuat model dengan K-Nearest Neighbor.
K-Nearest Neighbor adalah algoritma supervised learning dimana hasil dari instance yang baru diklasifikasikan berdasarkan mayoritas dari kategori k-tetangga terdekat. Tujuan dari algoritma ini adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data. Algoritma k-Nearest Neighbor menggunakan Neighborhood Classification sebagai nilai prediksi dari nilai instance yang baru.
Kelebihan yang dimiliki KNN antara lain:
-- K-NN cukup intuitif dan sederhana: Algoritma K-NN sangat sederhana untuk dipahami dan juga mudah diimplementasikan. Akurasi tinggi (relatif).
-- K-NN tidak memiliki asumsi: K-NN adalah algoritma non-parametrik yang berarti ada asumsi yang harus dipenuhi untuk mengimplementasikan K-NN. Model parametrik seperti regresi linier memiliki banyak asumsi yang harus dipenuhi oleh data sebelum dapat diimplementasikan yang tidak terjadi pada K-NN.
-- Dapat digunakan baik untuk Klasifikasi dan Regresi: Salah satu keuntungan terbesar K-NN adalah bahwa K-NN dapat digunakan baik untuk masalah klasifikasi dan regresi. Karena kasus pada data yang digunakan termasuk kasus hubungan pengaruh variabel prediktor terhadap variabel hasil, maka data tersebut cocok menggunakan metode K-NN.
Kekurangan yang dimiliki KNN yakni algoritme menjadi lebih lambat secara signifikan karena jumlah contoh dan/atau prediktor/variabel yang meningkat.

- Menggunakan cosine similarity. Algoritma ini dipilih karena mudah digunakan dan juga sebagai pembanding dengan sistem rekomendasi dengan model. Cosine similarity singkatnya digunakan untuk mengukur kemiripan antara dua buah vektor dan kesamaan arahnya dengan cara menghitung sudut kosinus dari kedua vektornya.
## Data Understanding
![2](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/dataset.jpg)
 [Dataset](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps) yang saya gunakan berasal dari [Kaggle](https://www.kaggle.com/). Dengan informasi sebagai berikut:
![3](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/infoTable.jpg) 
 - Jenis dan ukuran berkas : CSV(14 MB)
 -  Memiliki 17 kolom.
 -  Memiliki 7197 row data.

Variabel-variabel pada Dataset dataset adalah sebagai berikut :
1. "id" merupakan kolom data unik id dari aplikasi.
2. "track_name" merupakan kolom dengan data nama dari aplikasi.
3. "size_bytes" merupakan kolom dengan data ukuran dari aplikasi dalam satuan byte.
4. "currency" merupakan kolom dengan data mata uang dari aplikasi dalam satuan dollar (USD).
5. "price": merupakan kolom dengan data harga dari aplikasi dalam satuan dollar.
6. "ratingcounttot" merupakan  kolom dengan data penilaian pengguna mengenai semua versi dari aplikasi dalam satuan bintang.
7. "ratingcountver" merupakan  kolom dengan data penilaian pengguna mengenai versi saat ini dari aplikasi dalam satuan bintang.
8. "user_rating" merupakan  kolom dengan data rata-rata penilaian pengguna mengenai semua versi ini dari aplikasi dalam satuan bintang.
9. "userratingver" merupakan  kolom dengan data rata-rata penilaian pengguna mengenai versi saat ini dari aplikasi dalam satuan bintang.
10. "ver" merupakan kolom kode versi aplikasi yang digunakan saat ini.
11. "cont_rating"merupakan kolom dengan data kategori usia penggunaan untuk aplikasi, seperti 17+.
12. "prime_genre" merupakan kolom dengan data kategori dari genre aplikasi. 
13. "sup_devices.num" merupakan kolom dengan nomor supporting devices. Supporting devices adalah perangkat tambahan atau pendukung.
14. "ipadSc_urls.num" merupakan kolom dengan data jumlah screenshot yang ditampilkan pada aplikasi saat akan diunduh.
15. "lang.num" merupakan kolom dengan data jumlah bahasa yang tersedia di aplikasi.
16. "vpp_lic"merupakan kolom lisensi berbasis perangkat Vpp yang diaktifkan.

Berikut merupakan data yang divisualisasikan setelah gambar dibersihkan :
![4](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/visualisasi.png)
Dari visualisasi diatas, kita dapat melihat genre yang paling tinggi dengan kriteria berbayar dan tidak berbayar. Genre Games merupakan genre yang paling banyak memiliki aplikasi berbayar dan tidak berbayar.
## Data Preparation
Berikut tahap - tahap yang dilakukan pada data preparation :
- Memeriksa data yang bernilai null, kosong dan duplikat pada data. Pada tahap ini, data sudah bersih,  tidak memiliki data bernilai null, kosong dan duplikat yang harus dihapus.
- Menghapus kolom yang tidak diperlukan dengan menggunakan menggunakan method df.drop, mengganti nama kolom yang digunakan dengan method df.rename pada dataframe dari dataset, dan menyusun kolom dengan rapih menggunakan method df.loc[:, sehingga kolom yang digunakan sebagai berikut 'app_name', 'genre', 'user_rating_ver', 'version_rating', 'price', 'supp_devices', 'screen_shots_displayed', 'size_bytes'.
- Konversi label kategori menjadi one-hot encoding. Hal ini dilakukan untuk memudahkan pencarian nilai terdekat dari setiap aplikasi. Sebelumnya data ini merupakan data kategori dan dirubah menjadi data numerik yang ada pada setiap kolom yang berbeda kategori. Proses ini dialkukan menggunakan method get_dummies pada kolom dataframe dari dataset yang selanjutnya datanya disatukan pada dataframe.
- Tahap terakhir dengan melakukan standarisasi data. Jika salah satu variabel memiliki jarak yang dengan rentang yang jauh(skala berbeda), maka nilai tersebut harus disesuaikan. Jadi jarak antara semua variabel harus distandarisasi agar jarak akhir yang didapatkan proposional. Hal ini akan membuat semua fitur numerik berada dalam skala data yang sama juga membuat komputasi dari pembuatan model dapat berjalan lebih cepat karena rentang datanya hanya antara 0-1. Untuk melakukan standarisasi data, digunakan fungsi MidMaxScaler. 
## Modeling
Pada modeling ini, ada dua teknik yang digunakan (dan telah disinggung di bagian project overview di atas). Model yang digunakan untuk proyek kali ini adalah K-Nearest Neighbor dan Cosine Similarity untuk content-based filterting. Berikut adalah hasilnya :
1.  K-Nearest Neighbor
Untuk membangun model ini, digunakan fungsi  [NearestNeighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) dari sklearn dengan parameter metriksnya yakni euclidian. Kemudian fungsi tersebut di inisiasikan sebagai model yang selanjutnya dilakukan fitting terhadap data yang ada pada dataframe. Setelah itu dibuat fungsi getRecommendedApps_model untuk memberikan rekomendasi terhadap suatu nama aplikasi yang dimana skenarionya adalah apabila pengguna menyukai atau mengunduh aplikasi tersebut, maka berikan rekomendasi ini sebagai aplikasi yang mungkin disukai. Hasil rekomendasinya adalah seperti berikut :
![KNN](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/KNN.jpg)
2.  Cosine Similarity
Selanjutnya, rekomendasi pun dapat diberikan dengan menghitung cosine similarity dari setiap data di dataset menggunakan fungsi cosine_similarity dari sklearn. Prosesnya adalah dengan memanggil fungsi cosine_similarity dengan argumen dataframe sebagai objeknya. Kemudian hasil dari perhitungannya disimpan pada dataframe baru. Untuk tahapan pemberian rekomendasinya, dibuat fungsi getRecommendedApps_cosine dimana fungsi tersebut akan memberikan rekomendasi terhadap suatu nama aplikasi dengan skenario yang sama dengan nomor 1.
Pada fungsi tersebut, akan dilakukan pencarian kolom dari suatu nama aplikasi pada dataframe baru hasil perhitungan cosine similarity. Kemudian diurutkan nilainya berdasarkan nilai cosine similarity tertinggi dan juga urutannya. Setiap urutan ke 2 terakhir sampai ke n terakhir merupakan kandidat yang memiliki nilai cosine similarity yang sama maka akan ditampilkan sebagai hasil rekomendasinya. Urutan paling terakhir merupakan nilai cosine similarity dari kolom dengan nama aplikasi yang sama. Untuk lebih jelasnya hasil rekomendasi dapat dilihat seperti berikut ini :
![COSINE](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/COSINE.jpg)
## Evaluation
1. Pada K-Nearest Neighbor hasil yang didapatkan tergolong sudah sebesar 94.73% dengan menggunakan parameter metriksnya : euclidian.
![KNN-EUCLIDIAN](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/evalknn.jpg)
Perhitungan menggunakan euclidian sebagai berikut :
![rumus-euclidian](https://1.bp.blogspot.com/-42VvprfY9iA/XvZ-gi_7d0I/AAAAAAAAA-c/B9GUJEXZGHcC-W23rsU1qVmikoU7oWr8ACLcBGAsYHQ/s1600/Rumus-Euclid.jpg)
 Jarak euclidian adalah yang paling banyak digunakan karena merupakan metrik default yang digunakan pustaka Python SKlearn untuk K-Tetangga Terdekat. Ini adalah ukuran jarak garis lurus antara dua titik di ruang Euclidean. 
Untuk mengukur kinerja model KNN untuk sistem rekomendasi digunakan beberapa metriks diantaranya :
 - Calinski-Harabasz
 Apabila kita mencoba menggunakan Calinski-Harabasz, diperoleh hasil sebagai berikut :
![Perolehan-Calinski](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/evalch.jpg)
Skor Calinski Harabasz digunakan untuk menghitung kriteria rasio varian. Metriks ini digunakan pada model clustering seperti yang saat ini sedang digunakan. Skor semakin tinggi ketika cluster padat dan terpisah dengan baik. Pada model ini, nampaknya cluster masih belum padat dan terpisahkan dengan baik.  Hasil perolehan dapat dihitung dengan menggunakan scikit-learn dengan cara berikut:
![rumus-Calinski](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/pyc.jpg)
 - Davies-Bouldin
 Apabila kita mencoba menggunakan Calinski-Harabasz, diperoleh hasil sebagai berikut :
![perolehan-Harabasz](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/evald.jpg)
Skor Davies Bouldin digunakan untuk menilai separasi tiap cluster dari model. Metriks ini digunakan pada model clustering seperti yang saat ini sedang digunakan. Skor rendah ketika separasi tiap cluster di model terpisahkan dengan baik. Pada model ini skornya cukup kecil sehingga menandakan modelnya sudah memiliki separasi cluster yang baik.  Hasil perolehan dapat dihitung dengan menggunakan scikit-learn dengan cara berikut:
![rumus-Harabasz](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/pyd.jpg)
2. Pada proyek menggunakan algoritma cosine similarity, teknik perhitungan yang dilakukan menggunakan formula berikut : 
![rumus-cos](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/cos.jpg)
Dengan menggunakan perhitungan diatas diperoleh tingkat rekomendasi yang sangat baik, dengan tingkat kesamaan 99% berikut :
![COSINE-KESAMAAN](https://raw.githubusercontent.com/farischamakay/ML5---Prediksi-pasien-diabetes/main/SubmissionAkhir/evalcs.jpg)
Nilai akurasi klasifikasi KNN yang menggunakan cosine similarity lebih tinggi  dari pada euclidean distance, disebabkan oleh ketelitian pengukuran sudut vektor dokumen. Pengukuran similaritas yang lebih efektif untuk klasifikasi dokumen teks  adalah menggunakan pengukuran sudut cosinus seperti pada cosine similarity. Pada pengukuran cosine similarity, kesamaan dokumen tidak terlalu dipengaruhi oleh panjang dokumen. Pengukuran dengan nilai cosine similarity lebih tinggi menunjukkan nilai similaritas lebih tinggi dibandingkan dengan nilai cosine similarity lebih rendah.







