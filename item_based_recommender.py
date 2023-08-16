###########################################
# Item-Based Collaborative Filtering
###########################################

# Öğe tabanlı işbirlikçi filtreleme, kullanıcılara kendi tercihlerine ve benzer kullanıcıların tercihlerine göre
# kişiselleştirilmiş öneriler sunmak için tavsiye sistemlerinde kullanılan bir tekniktir. Kullanıcılardan ziyade öğeler
# arasındaki benzerliğe odaklanan bir işbirlikçi filtreleme biçimidir.

# Öğe tabanlı işbirlikçi filtrelemede öneriler, bir kullanıcının daha önce ilgi gösterdiği öğelere benzer öğeler
# belirlenerek oluşturulur. Bunun altında yatan varsayım, bir kullanıcının belirli bir öğeyi beğenmesi veya onunla
# etkileşime girmesi halinde, diğer benzer öğeler için de benzer tercihlere sahip olma ihtimalinin yüksek olduğudur.

# Öğe tabanlı işbirlikçi filtreleme süreci tipik olarak aşağıdaki adımları içerir:

# Veri toplama: Derecelendirmeler, incelemeler veya satın alma geçmişi gibi kullanıcı-öğe etkileşimleri hakkında veri
# toplama.

# Öğe benzerlik hesaplaması: Kosinüs benzerliği veya Pearson korelasyonu gibi çeşitli metriklere dayalı olarak öğeler
# arasındaki benzerliği hesaplayın. Benzerlik genellikle her iki öğe ile etkileşime giren kullanıcıların
# derecelendirmeleri veya tercihleri karşılaştırılarak belirlenir.

# Komşuluk seçimi: Sistemdeki her bir öğe için benzer öğelerden oluşan bir alt küme belirleyin. Öğenin komşuluğu olarak
# bilinen bu alt küme, söz konusu öğeye en çok benzeyen öğelerden oluşur.

# Öneri oluşturma: Öğenin komşuluğu oluşturulduktan sonra, sistem benzer kullanıcıların tercihlerini dikkate alarak
# öneriler oluşturabilir. Belirli bir kullanıcı için sistem, kullanıcının etkileşime girmediği komşuluğundaki öğeleri
# tanımlar ve kullanıcının muhtemelen ilgileneceği varsayımına dayanarak bu öğeleri önerir.

# Öğe tabanlı işbirlikçi filtrelemenin çeşitli avantajları vardır. Hesaplama açısından verimlidir ve büyük veri kümeleri
# ve öğe kataloglarıyla başa çıkabilir. Ayrıca, yeni kullanıcılar veya öğeler hakkında sınırlı bilginin olduğu "soğuk
# başlangıç" sorunuyla uğraşırken de iyi performans gösterir. Ek olarak, öğe benzerliklerine dayalı doğru öneriler
# sağlayabilir.

# Bununla birlikte, öğe tabanlı işbirlikçi filtreleme, kullanıcı-öğe etkileşim matrisinin seyrek olduğu "seyreklik"
# probleminden muzdarip olabilir, bu da çoğu kullanıcının mevcut öğelerin yalnızca küçük bir kısmıyla etkileşime girdiği
# anlamına gelir. Bu gibi durumlarda, tavsiye için yeterli sayıda benzer öğe bulmak zor olabilir.

# Genel olarak, öğe tabanlı işbirlikçi filtreleme, özellikle öğe benzerliklerinin iyi tanımlandığı ve kolayca
# hesaplandığı senaryolarda, tavsiye sistemleri oluşturmada popüler ve etkili bir yaklaşımdır.


#########################################
# İş Problemi
#########################################

# Online bir film izleme platformu iş birlikçi filtreleme yöntemi ile bir öneri sistemi geliştirmek istiyor.
# İçerik temelli öneri sistemlerini deneyen şirket, topluluğun kanaatlerini barındıracak şekilde öneriler geliştirmek
# istiyor.

# Kullanıcılar bir filmi beğendiğinde o film ile benzer beğenilme örüntüsüne sahip olan diğer filmler önerilmek
# istenmektedir.


##########################################
# Veri Setinin Hikayesi
##########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Veri seti MovieLens  isimli bir firma tarafından sağlanmıştır.
# İçerisinde filmler ve bu filmlere verilen puanları barındırmaktadır.
# Veri seti yaklaşık 27000 film için yaklaşık 2000000 derecelendirme içermektedir.

# Veri seti iki csv dosyasından oluşmaktadır.
# 1. csv dosyası : movie.csv dosyası
# - movield = Eşsiz film numarası
# - title = Film adı

# 2. csv dosyası : rating.csv dosyası
# - userid = Eşsiz kullanıcı numarası.
# - movield = Eşsiz film numarası
# - rating = Kullanıcı tarafından filme verilen puan
# - timestamp = Değerlendirme tarihi

###################################
# Yol Haritası
###################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") # iki csv dosyası merge ile mavie solda rating sağda olacak şekilde
# movieId ye göre birleştirildi.
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.head()
df.shape

df["title"].nunique()

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





