import re

# Daftar prefix plat nomor Indonesia
plat_nomor_indonesia = [
    "A", "B", "D", "E", "F", "G", "H", "I", "J", "K", 
    "L", "M", "N", "P", "R", "S", "T", "Z", "AB", "BB", 
    "BD", "BG", "BH", "BK", "BL", "BM", "BN", "BP", "BR", 
    "BS", "BT"
]

# Data yang akan diperiksa
data = ["BG8110CE", "07 . 2 7"]

# Fungsi untuk memeriksa apakah sebuah string adalah plat nomor
def is_plat_nomor(plat):
    # Menghapus spasi ekstra
    plat = plat.replace(" ", "")  
    # Regex untuk mencocokkan plat nomor dengan format yang benar
    match = re.match(r"^([A-Z]{1,2})\d{1,4}[A-Z]{1,3}$", plat)
    if match:
        kode = match.group(1)
        return kode in plat_nomor_indonesia
    return False

# Memeriksa data
for plat in data:
    if is_plat_nomor(plat):
        print(f"{plat} adalah plat nomor")
    else:
        print(f"{plat} bukan plat nomor")
