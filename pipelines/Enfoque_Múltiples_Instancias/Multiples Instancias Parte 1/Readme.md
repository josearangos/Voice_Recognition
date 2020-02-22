Enfoque multiples instacias


### Features

1. Se obtiene los MFCCs de cada audio, usando 20 coeficientes y ventanas de 0.025 segundo.
2. Se hacen grupos de 10 ventanasy si el número de ventanas no es multiplo de 10, las sobrantes queda en un grupo, es decir: si hay 101 ventanas serían 10 grupos de 10 ventanas y un grupo de 1 ventana.
3. Se promedian los coeficientes por grupos.

Ejemplo: Si un audio tiene 108 ventanas, las matriz de instancias que representan este vector es de dimensiones (11,23)
