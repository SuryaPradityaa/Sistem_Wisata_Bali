import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset_wisata_bali.csv")

df['fitur'] = (
    df['kategori'] + " " +
    df['kabupaten'] + " " +
    df['aktivitas'] + " " +
    df['deskripsi']
)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['fitur'])

cosine_sim = cosine_similarity(tfidf_matrix)

def rekomendasi(nama_tempat, top_n=5):
    idx = df[df['nama_tempat'] == nama_tempat].index[0]
    
    skor = list(enumerate(cosine_sim[idx]))
    skor = sorted(skor, key=lambda x: x[1], reverse=True)
    
    hasil = skor[1:top_n+1]
    
    rekomendasi_list = []
    
    for i in hasil:
        data = df.iloc[i[0]]
        
        rekomendasi_list.append({
            "nama": data['nama_tempat'],
            "kategori": data['kategori'],
            "lokasi": data['kabupaten'],
            "rating": data['rating']
        })

    rekomendasi_list = sorted(rekomendasi_list, key=lambda x: x['rating'], reverse=True)

    return rekomendasi_list


if __name__ == "__main__":
    print(df["nama_tempat"].tolist())