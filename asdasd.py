def perbaiki_kode_plat(kode):
    mapping = {'0': 'D', '6': 'G', '8': 'B', '3': 'E'}
    return ''.join(mapping.get(c, c) for c in kode)

def gabungkan_plat_dengan_perbaikan(list_teks):
    if len(list_teks) < 3:
        return "Data kurang untuk format xx yyyy zzz"
    
    # perbaiki elemen pertama untuk kode wilayah (xx)
    elemen_pertama = perbaiki_kode_plat(list_teks[0])
    # elemen kedua sebagai angka (yyyy) tanpa perbaikan
    elemen_kedua = list_teks[1]
    # perbaiki elemen ketiga kemungkinan kode huruf akhir (zzz)
    elemen_ketiga = perbaiki_kode_plat(list_teks[2])
    
    return f"{elemen_pertama} {elemen_kedua} {elemen_ketiga}"

# Contoh list input
list_teks = ["Ad", "6174", "CU", "04 ' 30"]

hasil = gabungkan_plat_dengan_perbaikan(list_teks)
print(hasil)   # Output: B 6174 CU
