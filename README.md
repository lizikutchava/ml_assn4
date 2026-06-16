# Facial Expression Recognition Challenge

**Kaggle:** https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge

**W&B:** https://wandb.ai/lkuch23-free-university-of-tbilisi-/fer2013-challenge

## დავალება

48×48 პიქსელიანი შავ-თეთრი სურათების კლასიფიკაცია 7 ემოციად:

| ლეიბლი | ემოცია     |
|---------|------------|
| 0       | გაბრაზება  |
| 1       | ზიზღი     |
| 2       | შიში       |
| 3       | სიხარული  |
| 4       | სევდა      |
| 5       | გაკვირვება  |
| 6       | ნეიტრალური|

**მონაცემების გაყოფა:**
- `train.csv` → 80% train (22,967 ნიმუში) / 20% val (5,742 ნიმუში) - stratified split
- `test.csv` → submission-ის სეტი (7,178 ნიმუში)

**გადაწყვეტილება:** 
stratified split ავირჩიეთ, რადგან საჭირო იყო კლასების პროპორციების შენარჩუნება train და val სეტებში. FER2013 დაუბალანსებელია, Happy-ს Disgust-ზე დაახლოებით 3-ჯერ მეტი ნიმუში აქვს.

---

## რეპოზიტორიის სტრუქტურა

ml_assn4/

├── FER_Training.ipynb                      ← სატრენინგო notebook (Colab)

├── README.md                               ← ეს ფაილი

├── submission.csv                          ← Kaggle-ის მოლოდინები საუკეთესო მოდელიდან

├── exp2_overfit_curves.png                 ← train vs val მრუდები DeepCNN-ისთვის

├── exp3_regularisation_comparison.png      ← Exp2 vs Exp3 gap-ის შედარება

├── exp4_comparison.png                     ← ყველა ექსპერიმენტი MiniResNet-მდე

├── exp5_final_comparison.png               ← საბოლოო შედარება
└── .gitignore

---

## ექსპერიმენტები

5-ვე ექსპერიმენტი დალოგილია W&B-ზე სხვადასხვა run-ებად.
თითოეული run-ი იწერს: train/loss-ს, train/acc-ს, val/loss-ს, val/acc-ს, train_val_gap-ს, lr-ს, გრადიენტის ნორმებს, confusion matrix-სა და per-class F1-ს.

---

### ექსპერიმენტი 1 - TinyCNN (Underfitting)

**არქიტექტურა:**

Conv(1→8) → ReLU → MaxPool

Conv(8→16) → ReLU → MaxPool

Flatten → Linear(2304→64) → ReLU → Linear(64→7)

~25,000 პარამეტრი.

**გადაწყვეტილებები:**
- არ ვიყენებთ აუგმენტაციას, რათა ვნახოთ მოდელის შესაძლებლობა დახმარების გარეშე.
- არ ვიყენებთ Dropout-სა და BatchNorm-ს.
- Adam ოპტიმიზატორი, LR=1e-3, 30 ეპოქა, early stopping-ს გარეშე, ვრანავთ ბოლომდე, რათა სრული ვნახოთ სატრენინგო მრუდი.
- პატარა არქიტექტურა.

შედეგი: **Val acc = 0.5070**

**ანალიზი:** ეს მოდელი capacity-ის ჭერს ეჯახება. Val accuracy ჩერდება დაახლოებით მე-10 ეპოქაზე ~50%-ზე და შემდეგ 20 ეპოქაში თითქმის არ იცვლება. როგორც train, ასევე val loss მაღალია, ანუ გვაქვს underfitting.

---

### ექსპერიმენტი 2 - DeepCNN (Overfitting)

**არქიტექტურა:**

Conv(1→32) → ReLU → Conv(32→32) → ReLU → MaxPool

Conv(32→64) → ReLU → Conv(64→64) → ReLU → MaxPool

Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool

Flatten → Linear(4608→1024) → ReLU → Linear(1024→512) → ReLU → Linear(512→7)

~4,000,000 პარამეტრი.

**გადაწყვეტილებები:**
- 6 conv layer, წინა ექსპერიმენტზე ღრმაა.
- არ ვიყენებთ BatchNorm-ს, Dropout-სა და აუგმენტაციას.
- Weight decay = 0.0
- Early stopping არ გვაქვს.
- იგივე LR და ოპტიმაიზერი.

შედეგი: **Val acc = 0.5521**

**ანალიზი:** გვაქვს overfitting. Train accuracy აღწევს ~85-90%-ს, ხოლო val ჩერდება 50-55%-ზე.

---

### ექსპერიმენტი 3 - MediumCNN 

**არქიტექტურა:**

Conv(1→32) → BN → ReLU → MaxPool → Dropout2d(0.1)

Conv(32→64) → BN → ReLU → MaxPool → Dropout2d(0.1)

Conv(64→128) → BN → ReLU → MaxPool → Dropout2d(0.2)

Conv(128→256) → BN → ReLU → MaxPool

Flatten → Linear(2304→512) → BN → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→7)

~1,500,000 პარამეტრი.

**გადაწყვეტილებები:**
- ვიყენებთ BatchNorm-ს თითოეული conv-ის შემდეგ, რაც ტრენინგს ასტაბილურებს, წონებისადმი სენსიტიურობას ამცირებს, მაღალი LR-ს გამოყენების საშუალებას იძლევა.
- Dropout2d(0.1/0.2) conv block-ებში, შემთხვევით ანულებს feature map-ებს.
- Dropout(0.3) classifier-ში, FC layer-ებს ხელს უშლის კოადაპტაციაში.
- მონაცემთა აუგმენტაციაა ვიყენებთ, რენდომ ჰორიზონტალური ფლიპი (p=0.5), ±10° ბრუნვა, ±10% affine translate. სახის გამომეტყველება ჰორიზონტალურად სიმეტრიულია, შესაბამისად დათასეტი უფრო ეფექტური ხდება.
- Label smoothing = 0.1, არბილებს hard one-hot ლეიბლებს, regulariser-ივით იქცევა.
- AdamW Adam-ის ნაცვლად, AdamW weight decay-ს გრადიენტის განახლებისგან ანცალკევებს.
- Early stopping (patience=15), წყვეტს ტრენინგს თუ val acc 15 ეპოქის განმავლობაში არ გაუმჯობესდა.

შედეგი: **Val acc = 0.6569**

**ანალიზი:** მსგავსი სიღრმის მიუხედავად დიდი გაუმჯობესება Exp2-თან შედარებით. Train-val gap მნიშვნელოვნად შემცირდა. BatchNorm + Dropout ერთად უფრო ეფექტურია ვიდრე ცალ-ცალკე. აუგმენტაცია მოდელს ხელს უშლის კონკრეტული პიქსელის პატერნების დამახსოვრებაში.

---

### ექსპერიმენტი 4 - MiniResNet (Residual Connections)

**არქიტექტურა:**

Stem: Conv(1→32) → BN → ReLU

Layer1: 2× ResidualBlock(32→64,  stride=2)   → 24×24

Layer2: 2× ResidualBlock(64→128, stride=2)   → 12×12

Layer3: 2× ResidualBlock(128→256, stride=2)  → 6×6

Layer4: 1× ResidualBlock(256→512, stride=2)  → 3×3

GlobalAvgPool → Dropout(0.4) → Linear(512→7)

თითოეული ResidualBlock: `out = ReLU(BN(Conv(x)) + shortcut(x))`

~3,000,000 პარამეტრი.

**გადაწყვეტილებები:**
- Residual (skip) connections
- Global Average Pooling Flatten-ის ნაცვლად, ბოლო conv block-ის შემდეგ Flatten-ის ნაცვლად თითოეულ feature map-ს ერთ რიცხვამდე ვასაშუალოებთ, რითიც feature-ების რაოდენობა მცირდება, კლასიფიკატორში პარამეტრების რაოდენობას მნიშვნელოვნად ამცირებს და ძლიერ რეგულარიზაციას იძლევა.
- LR (5e-4 vs 1e-3), ღრმა ქსელები residual connection-ებით LR-ის მიმართ უფრო სენსიტიურია, ოდნავ პატარა LR სტაბილურ კონვერგენციას იძლევა.
- Dropout(0.4), წინა ექსპერიმენტზე დიდი, რადგან მოდელი ღრმავდება და overfitting-ის რისკი იზრდება.
- 60 ეპოქა, residual network-ებს მეტი ეპოქა ჭირდებათ.
  
შედეგი: **Val acc = 0.6853** (საუკეთესო შედეგი საბოლოოდ)

**ანალიზი:** MiniResNet-მა ყველა სხვა მოდელი გაუსწრო, მათ შორის pretrained TransferResNet18-ს. Residual Connections საშუალებას იძლევა ღრმა ქსელი ეფექტურად დავატრენინგოთ vanishing gradients-ის გარეშე. Global Average Pooling-ით classifier-ი გამარტივდა და overfitting-ის რისკი შემცირდა.

---

### ექსპერიმენტი 5 - Transfer Learning: ResNet18

**არქიტექტურა:**
წინასწარ გავარჯიშებული ResNet18 (ImageNet1K), FER2013-თვის ადაპტირებული:
conv1: შეცვლილია 1 არხის მისაღებად. წონები ინიციალიზებულია წინასწარ დატრენინგებული RGB წონების საშუალოთი, რაც edge/texture detector-ებს ინარჩუნებს ნულიდან დაწყების ნაცვლად.
fc: შეცვლილია Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→7)-ით

**გადაწყვეტილებები:**
- წინასწარ დატრენინგებული წონებ, ImageNet pretraining-ი მოდელს მდიდარ low-level feature detector-ებს აძლევს, რაც სახის ანალიზში გვეხმარება.
- RGB წონების საშუალო conv1-ისთვის, სტანდარტული მიდგომა RGB წინასწარ დატრენინგებული მოდელების grayscale-ზე ადაპტაციისთვის.
- LR (1e-4), fine-tuning-ს გაცილებით პატარა LR სჭირდება ვიდრე ნულიდან ტრენინგს. მაღალი LR გაანადგურებდა წინასწარ დასწავლილ feature-ებს.
- პატარა batch size (32 vs 64), ResNet18 უფრო დიდია და sample-ზე მეტ GPU მეხსიერებას იყენებს
- Early stopping (patience=12), მოკლე patience, რადგან fine-tuning სწრაფად მიდის ერთი წერტილისკენ.

შედეგი: **Val acc = 0.6513**

**ანალიზი (მოულოდნელი შედეგი):** Transfer learning-მა MiniResNet-ს ვერ გაუსწრო (0.6513 vs 0.6853). სავარაუდოდ იმის გამოა, რომ ImageNet pretraining ხდება მაღალი გარჩევადობის (224×224) RGB ბუნებრივ სურათებზე, ხოლო FER2013-ის სურათები დაბალი გარჩევადობის (48×48) grayscale სახეებია.

---

## W&B Tracking

ყველა ექსპერიმენტი: https://wandb.ai/lkuch23-free-university-of-tbilisi-/fer2013-challenge

თითოეულ ეპოქაზე დალოგილი მეტრიკები:
- `train/loss`, `train/acc`
- `val/loss`, `val/acc`
- `train_val_gap` - overfitting-ის მთავარი ინდიკატორი
- `lr`
- `gradients/{layer}_norm` - per-layer გრადიენტის ნორმები (ყოველ 5 ეპოქაში)
- `confusion_matrix` - სურათი ილოგება თითოეული run-ის ბოლოს
- `per_class/{emotion}_f1` - F1 სქორი თითოეული ემოციის კლასისთვის
