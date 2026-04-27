# Rapport: Assignment 1 Part 2

## TL;DR

Jag tränade en handfull modeller på MNIST och jämförde hur de presterade när jag stegvis bytte från en enkel FFN till olika CNN-varianter och sen lade på regularization. Den korta versionen är att FFN-baseline landade på **97.64%** test accuracy, och så fort jag bytte till en CNN hoppade resultatet upp till nästan 99%. När jag lade på dropout, batch normalization och weight decay på 3-layer CNN:en kom jag till **99.21%**, och med lite hyperparameter-tuning ovanpå det pressades det vidare till **99.34%** ("low_noise"-konfigen).

Det jag tycker är mest värt att lyfta är inte slutsiffran utan att varje körning är spårbar: alla körningar har en egen `outputs/run_<timestamp>/`-mapp med config, checkpoints, träningskurvor, confusion matrix och TensorBoard-loggar. Det betyder att jag kan gå tillbaka månader senare och faktiskt förstå vad jag gjorde, inte bara veta vad det landade på.

![Sammanfattning av alla körningar](outputs/summary_all_runs.png)

## Inledning

Målet med den här delen var att bygga vidare från en enkel feed-forward-modell och se vad som händer när jag byter till mer bildanpassade arkitekturer. Samtidigt ville jag börja jobba mer strukturerat — alltså inte bara skriva en `.py`-fil och få ut en siffra, utan faktiskt logga, spara checkpoints och kunna jämföra körningar i efterhand.

Allt utgår från MNIST-datasetet. Modellerna är skrivna i PyTorch och loggas i TensorBoard. För varje körning sparar jag config, checkpoints, kurvor, confusion matrix och exempelbilder så jag kan granska träningen i detalj efteråt.

## Mål

Det jag ville få ut av del 2 var:

- en enklare MLOps-rutin som faktiskt fungerar för mig
- tydligare loggning av träning och prestanda
- testa vad data augmentation gör i praktiken
- byta från FFN till CNN och se hur stor skillnad det gör
- jämföra olika CNN-arkitekturer mot varandra på lika villkor

## MLOps och spårbarhet

För att slippa hamna i ett läge där jag har siffror men inte vet hur jag fick fram dem skapade jag en egen output-mapp per körning: `outputs/run_<timestamp>/`. I varje mapp sparas:

- `training_config.json` (alla hyperparametrar)
- `best.pt` (bästa checkpoint enligt val-loss)
- `epoch_NNN.pt` per epoch
- `curves_loss_acc.png`
- `confusion_matrix_percent.png`
- `examples_correct.png` och `examples_incorrect.png`
- TensorBoard-loggar i `tensorboard/`

I praktiken gör det här att jag kan följa träningen steg för steg, gå tillbaka och titta på en gammal körning, och välja bästa checkpoint utifrån val-loss istället för att bara ta sista epoken. Jag loggar också train/val loss, accuracy, tid per epoch, total träningstid, throughput och en augmentation-preview till TensorBoard, så jag kan se hur det rör sig live medan körningen pågår.

## Modell 1: Baseline FFN

Det jag började med var en enkel feed-forward, alltså en där bilden flattenas från `28x28` till en vektor på 784 värden och skickas genom två fully connected layers:

- `Flatten`
- `Linear(784, 128)`
- `ReLU`
- `Linear(128, 10)`

Den fungerar förvånansvärt bra på MNIST, men det är ganska tydligt att den inte tar vara på att indatan faktiskt är en bild — den ser bara 784 oberoende pixlar. Det är därför jag förväntade mig att en CNN skulle göra bättre ifrån sig.

## Data augmentation

För att se om jag kunde göra modellen mer robust testade jag data augmentation, men bara på train-setet. Validation och test fick vara orörda, annars blir utvärderingen orättvis. Jag använde:

- liten rotation
- liten förskjutning
- liten skalning
- liten shear
- Gaussian noise

Tanken är inte att göra bilderna *bättre*, utan att skapa rimliga variationer så att modellen inte memorerar exakt hur varje träningsexempel ser ut. För att få en känsla för vad det faktiskt gör loggade jag en preview till TensorBoard som visar originalbilden bredvid flera augmenterade versioner av samma siffra. Det blev mycket lättare att tolka när jag kunde se augmentation visuellt istället för bara läsa parametrarna.

## Regularization

Regularization handlar i grunden om att tvinga modellen att inte överanpassa sig till träningsdatan. Jag testade tre varianter parallellt: dropout, batch normalization och weight decay.

### Dropout

Dropout stänger av slumpmässiga neuroner under träningen. Effekten blir att modellen inte får luta sig på en enda väg genom nätet — den måste sprida ut sin kunskap. Jag tycker det enklaste sättet att tänka på det är att modellen tvingas träna utan vissa av sina "favoritneuroner" varje steg, och då blir hela nätet mer robust.

### Batch normalization

Batch normalization håller aktiveringarna mer stabila mellan lagren. Det gör i praktiken att träningen blir jämnare och ofta snabbare, och i många experiment ger det också en viss regularization-effekt på köpet. I CNN:erna lade jag den direkt efter conv-lagren.

### Weight decay

Weight decay är ett straff på stora vikter, så att optimeringen inte tillåts springa iväg till extrema parameter-värden. Det är ett ganska billigt sätt att uppmuntra enklare lösningar.

### Vad jag förväntade mig

Om regularization fungerar som tänkt brukar man se att train-resultatet kanske blir lite *sämre* (modellen får inte memorera lika bra), men val/test blir bättre eller mer stabilt. Skillnaden mellan train och val krymper. Det är samma sak jag tittade efter i mina körningar.

## CNN: varför convolutional layers?

Nästa steg var att byta ut början av modellen mot convolutional layers. En CNN passar bilder bättre eftersom den kan plocka upp lokala mönster — kanter, linjer, kurvor, delar av siffror — och eftersom den är translationsinvariant. Alltså modellen ska kunna känna igen samma siffra även om den ligger lite olika i bilden, vilket en flat FFN aldrig riktigt fattar.

Efter conv-lagren körde jag `ReLU` följt av `MaxPool2d`. Pooling minskar dimensionerna och hjälper modellen att fokusera på de features som faktiskt betyder något.

## CNN-arkitekturer som testades

Jag jämförde fem upplägg:

### 1. FFN-baseline

En enkel dense-modell utan conv-lager, samma som ovan.

### 2. FFN med augmentation

Samma grundmodell, men med augmentation på train-setet. Jag ville se om enbart augmentation kunde lyfta en FFN.

### 3. 2-layer CNN

```
Conv2d(1, 16, kernel_size=3, padding=1)
Conv2d(16, 32, kernel_size=3, padding=1)
MaxPool2d(2, 2) efter conv-lagren
```

### 4. 3-layer CNN

En djupare variant:

```
Conv2d(1, 8, kernel_size=3, padding=1)
Conv2d(8, 16, kernel_size=3, padding=1)
Conv2d(16, 32, kernel_size=3, padding=1)
```

Tanken var att se om ett extra conv-lager ger märkbart bättre features eller om det är overkill för MNIST.

### 5. 3-layer CNN med regularization

Samma 3-layer CNN, men med:

- `BatchNorm2d` efter varje conv-lager
- `Dropout` i fully connected-delen
- `weight_decay` i `Adam`-optimizern

Det är den modell jag förväntade mig skulle prestera bäst, och den gjorde också det.

## Features i tidiga och senare conv-lager

Det jag tycker är intressant med CNN:er är att de tidiga conv-lagren brukar lära sig riktigt enkla saker — vertikala och horisontella linjer, diagonaler, enkla kurvor — medan de senare lagren bygger vidare och fångar mer meningsfulla strukturer som hörn, slingor, eller delar av en hel siffra. För MNIST är det inte så mycket att hämta i djupare lager (siffrorna är ju ganska enkla), men det är samma princip som gör att djupare nät skalar så bra på mer komplex bilddata.

## Resultat

Här är de fem huvudkörningarna jag använde som jämförelse, alla med 5 epochs:

| Modell | Run | Best epoch | Test loss | Test accuracy |
| --- | --- | ---: | ---: | ---: |
| FFN baseline | `run_20260422_145158` | 5 | 0.0802 | 0.9764 |
| FFN + augmentation | `run_20260422_145249` | 5 | 0.1016 | 0.9698 |
| 2-layer CNN + augmentation | `run_20260422_145403` | 5 | 0.0305 | 0.9895 |
| 3-layer CNN + augmentation | `run_20260422_145458` | 5 | 0.0296 | 0.9895 |
| 3-layer CNN + regularization | `run_20260422_151947` | 5 | 0.0217 | 0.9921 |

## Tolkning av resultaten

Det tydligaste mönstret är att båda CNN-modellerna var klart bättre än FFN-varianterna, vilket var precis det jag förväntade mig. Det jag inte riktigt förväntade mig var att augmentation faktiskt gjorde FFN:en *sämre* i den här körningen — den landade på 96.98% mot baseline 97.64%. Min tolkning är att en så enkel FFN inte har kapacitet att hantera den extra variationen, och att augmentation kommer till sin rätt först när modellen är tillräckligt uttrycksfull för att dra nytta av den.

Några andra observationer:

- 2-layer CNN gav ett stort hopp uppåt jämfört med FFN, vilket bekräftar att MNIST verkligen är ett bildproblem och inte bara ett tabulärt klassificeringsproblem.
- 3-layer CNN landade på exakt samma test accuracy som 2-layer CNN, men med marginellt lägre test loss. Skillnaden är så liten att jag inte tycker man kan säga att det djupare nätet är "bättre" för just MNIST.
- 3-layer CNN med regularization blev bäst på alla mått: lägre test loss (0.0217) och högre test accuracy (99.21%) än alla andra. Det är där dropout, batch norm och weight decay tillsammans verkligen syntes.

Sammanfattat: CNN slår FFN, ett extra conv-lager hjälper marginellt, och regularization gör en konkret skillnad även när modellen redan presterar bra.

## Begränsningar i jämförelsen

Jag vill vara ärlig med att jämförelsen inte är perfekt kontrollerad. Ett par saker som gör att jag tar slutsatserna med en nypa salt:

- baseline-FFN och CNN-varianterna har inte exakt samma träningsupplägg
- augmentation användes inte i alla modeller
- vissa körningar har olika konfig som kanske påverkar utfallet

En mer rättvis jämförelse skulle vara att köra FFN och CNN med exakt samma upplägg, både med och utan augmentation, och hålla epochs och övriga hyperparametrar identiska. Det är något jag skulle göra om om jag hade mer tid.

## Exempel på sparade artefakter

Några av de bilder som skapas automatiskt per körning:

### Baseline FFN: träningskurvor

![Baseline curves](outputs/run_20260422_145158/curves_loss_acc.png)

### 2-layer CNN: träningskurvor

![2-layer CNN curves](outputs/run_20260422_145403/curves_loss_acc.png)

### 3-layer CNN: confusion matrix

![3-layer CNN confusion matrix](outputs/run_20260422_145458/confusion_matrix_percent.png)

### 3-layer CNN: felklassificerade exempel

![3-layer CNN incorrect examples](outputs/run_20260422_145458/examples_incorrect.png)

### 3-layer CNN med regularization: träningskurvor

![Regularization curves](outputs/run_20260422_151947/curves_loss_acc.png)

## Överfitting

Jag använde val-loss för att avgöra vilken checkpoint som var "bäst" och sparade den som `best.pt`. Anledningen är att sista epoken inte alltid är den bästa: om train loss fortsätter ner men val loss börjar gå upp, har modellen börjat överanpassa sig. Genom att istället ta epoken med lägst val-loss får jag en mer robust modell för testet i slutet.

## Slutsats

Det jag tar med mig från det här är att CNN slår en enkel FFN ganska tydligt på MNIST, att augmentation hjälper när modellen är tillräckligt uttrycksfull men kan stjälpa när den inte är det, och att regularization (dropout + batch norm + weight decay tillsammans) ger en konkret förbättring även när basmodellen redan är bra.

Men det viktigaste är inte själva slutsiffran. Det viktigaste är att jag faktiskt kan reproducera vad jag gjorde. Jag kan öppna vilken körning som helst och se exakt vilken config som användes, vilka kurvor som ritades, vilka exempel som klassificerades fel. Det är en enorm skillnad jämfört med hur jag normalt skulle tackla en sån här uppgift, och jag tror det är den största lärdomen från Part 2 — att ha en arbetsprocess som inte producerar "siffror utan minne".

## Möjligt fortsatt arbete

Om jag skulle bygga vidare hade jag velat:

- köra FFN och CNN med exakt samma träningsupplägg för en mer kontrollerad jämförelse
- testa dropout och batch normalization separat för att se vilken som faktiskt drar lasset
- jämföra modeller med samma antal parametrar istället för samma antal lager
- logga sammanfattande metrics i en separat JSON så det blir lättare att aggregera över körningar
- visualisera vikter eller feature maps från conv-lagren — det vore intressant att faktiskt *se* vad de tidiga lagren har lärt sig
