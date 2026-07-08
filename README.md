# Sleep Health and Lifestyle Analysis

Ten projekt analizuje dane dotyczące snu, stylu życia i wybranych wskaźników zdrowotnych.

## Cel projektu

Celem było sprawdzenie, które czynniki mogą być powiązane z jakością snu i występowaniem zaburzeń snu.

Projekt zawiera:
- czyszczenie danych,
- analizę eksploracyjną,
- wizualizacje,
- analizę korelacji,
- prosty model uczenia maszynowego.

## Dane

Zbiór danych zawiera 374 obserwacje i 13 zmiennych.

W danych znajdują się informacje takie jak:

- płeć,
- wiek,
- zawód,
- długość snu,
- jakość snu,
- poziom aktywności fizycznej,
- poziom stresu,
- kategoria BMI,
- ciśnienie krwi,
- tętno,
- dzienna liczba kroków,
- zaburzenie snu.

Brakujące wartości w kolumnie `sleep_disorder` potraktowałam jako brak zgłoszonego zaburzenia snu.

## Narzędzia

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## Etapy analizy

- Wczytanie danych
- Uporządkowanie nazw kolumn
- Obsługa brakujących wartości
- Utworzenie dodatkowych zmiennych
- Analiza jakości i długości snu
- Analiza poziomu stresu, BMI i aktywności fizycznej
- Przygotowanie wykresów
- Sprawdzenie korelacji między zmiennymi
- Zbudowanie prostych modeli klasyfikacyjnych

## Najważniejsze wnioski

- Osoby bez zgłoszonych zaburzeń snu miały najwyższą średnią jakość snu.
- Osoby z bezsennością miały najniższą średnią jakość snu.
- Osoby z bezsennością spały średnio krócej niż pozostałe grupy.
- Wyższy poziom stresu był powiązany z niższą jakością snu.
- Długość snu była silnie dodatnio powiązana z jakością snu.
- Model regresji logistycznej osiągnął najlepszy wynik klasyfikacji.
- Accuracy najlepszego modelu wyniosło około 90%.

## Wizualizacje

### Rozkład jakości snu

![Rozkład jakości snu](images/sleep_quality_distribution.png)

### Długość snu a jakość snu

![Długość snu a jakość snu](images/sleep_duration_vs_quality.png)

### Poziom stresu a jakość snu

![Poziom stresu a jakość snu](images/stress_vs_sleep_quality.png)

### Jakość snu według kategorii BMI

![Jakość snu według kategorii BMI](images/sleep_quality_by_bmi.png)

### Aktywność fizyczna a jakość snu

![Aktywność fizyczna a jakość snu](images/physical_activity_vs_sleep_quality.png)

### Macierz korelacji

![Macierz korelacji](images/correlation_matrix.png)

### Rozkład zaburzeń snu

![Rozkład zaburzeń snu](images/sleep_disorder_distribution.png)

### Zaburzenia snu według kategorii BMI

![Zaburzenia snu według kategorii BMI](images/sleep_disorder_by_bmi.png)

### Jakość snu według zawodu

![Jakość snu według zawodu](images/sleep_quality_by_occupation.png)

### Poziom stresu według zawodu

![Poziom stresu według zawodu](images/stress_level_by_occupation.png)

### Tętno a poziom stresu

![Tętno a poziom stresu](images/heart_rate_vs_stress.png)

### Macierz pomyłek — regresja logistyczna

![Macierz pomyłek — regresja logistyczna](images/confusion_matrix_regresja_logistyczna.png)

### Macierz pomyłek — Random Forest

![Macierz pomyłek — Random Forest](images/confusion_matrix_random_forest.png)

### Najważniejsze cechy w modelu Random Forest

![Najważniejsze cechy w modelu Random Forest](images/feature_importance_random_forest.png)

## Model uczenia maszynowego

W projekcie zbudowałam dwa proste modele klasyfikacyjne:

- regresję logistyczną,
- Random Forest.

Celem modelu było przewidzenie, czy dana osoba ma zgłoszone zaburzenie snu.

Najlepszy wynik uzyskała regresja logistyczna.  
Model osiągnął accuracy na poziomie około 90%.

