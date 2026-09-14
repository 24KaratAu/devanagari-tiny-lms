# Synthetic reasoning corpus - hindi (sample)

Drawn from `test.jsonl`, so every name below is from the held-out entity pool.
Full corpus on Google Drive; regenerate with `python -m src.reasoning.generate --language hindi ...`.

## `rel2_which` - 2 entities, relative premises, 0 transitive hops

```
प्रश्न: भरत के अंक हिमांशु से ज़्यादा हैं।
हिमांशु और भरत में किसके अंक अधिक हैं?
तर्क: भरत > हिमांशु
उत्तर: भरत
```
*attribute* `marks` - *phrasing* `alt_lexical`/`which` - **held-out phrasing**

```
प्रश्न: स्वाति का वज़न मनीष से कम है।
मनीष और स्वाति में किसका वज़न अधिक है?
तर्क: मनीष > स्वाति
उत्तर: मनीष
```
*attribute* `weight` - *phrasing* `inverted`/`which`

## `val2_which` - 2 entities, values premises, 0 transitive hops

```
प्रश्न: नितिन की ऊँचाई १४९ सेंटीमीटर है। अरुण की ऊँचाई १७८ सेंटीमीटर है।
नितिन और अरुण में किसकी ऊँचाई अधिक है?
तर्क: अरुण = १७८, नितिन = १४९
उत्तर: अरुण
```
*attribute* `height` - *phrasing* `plain`/`which`

```
प्रश्न: भरत के अंक २७ हैं। हिमांशु के अंक ९० हैं।
हिमांशु और भरत में किसके अंक कम हैं?
तर्क: हिमांशु = ९०, भरत = २७
उत्तर: भरत
```
*attribute* `marks` - *phrasing* `plain`/`which`

## `val2_relation` - 2 entities, values premises, 0 transitive hops

```
प्रश्न: कुरता की कीमत १८८५ रुपये है। पंखा की कीमत ४२० रुपये है।
कुरता की कीमत पंखा से अधिक है, कम है या बराबर?
तर्क: कुरता = १८८५, पंखा = ४२०
उत्तर: अधिक
```
*attribute* `price` - *phrasing* `plain`/`relation`

```
प्रश्न: पंखा की लम्बाई १८ सेंटीमीटर है। कुरता की लम्बाई १८ सेंटीमीटर है।
कुरता की लम्बाई पंखा से अधिक है, कम है या बराबर?
तर्क: पंखा = १८, कुरता = १८
उत्तर: बराबर
```
*attribute* `length` - *phrasing* `plain`/`relation`

## `rel3_extreme` - 3 entities, relative premises, 1 transitive hop

```
प्रश्न: बबीता का वज़न सुधा से अधिक है। तरुण का वज़न बबीता से अधिक है।
किसका वज़न सबसे अधिक है?
तर्क: तरुण > बबीता, बबीता > सुधा
उत्तर: तरुण
```
*attribute* `weight` - *phrasing* `plain`/`genitive`

```
प्रश्न: बबीता का वज़न भरत से कम है। मनीष का वज़न बबीता से कम है।
किसका वज़न सबसे अधिक है?
तर्क: भरत > बबीता, बबीता > मनीष
उत्तर: भरत
```
*attribute* `weight` - *phrasing* `inverted`/`genitive`

## `rel3_pairwise` - 3 entities, relative premises, 1 transitive hop

```
प्रश्न: पंखा की लम्बाई दरी से अधिक है। दरी की लम्बाई साइकिल से अधिक है।
पंखा की लम्बाई साइकिल से अधिक है, कम है या बराबर?
तर्क: पंखा > दरी, दरी > साइकिल
उत्तर: अधिक
```
*attribute* `length` - *phrasing* `plain`/`relation`

```
प्रश्न: अरुण के अंक कल्पना से अधिक हैं। प्रवीण के अंक अरुण से अधिक हैं।
प्रवीण के अंक कल्पना से अधिक हैं, कम हैं या बराबर?
तर्क: प्रवीण > अरुण, अरुण > कल्पना
उत्तर: अधिक
```
*attribute* `marks` - *phrasing* `plain`/`relation`

## `val3_extreme` - 3 entities, values premises, 0 transitive hops

```
प्रश्न: करण की उम्र ४६ वर्ष है। शालिनी की उम्र ३३ वर्ष है। सुधा की उम्र ४१ वर्ष है।
किसकी उम्र सबसे अधिक है?
तर्क: करण = ४६, सुधा = ४१, शालिनी = ३३
उत्तर: करण
```
*attribute* `age` - *phrasing* `plain`/`genitive`

```
प्रश्न: तनुजा की ऊँचाई १२२ सेंटीमीटर है। सुधा की ऊँचाई १८९ सेंटीमीटर है। सचिन की ऊँचाई १३४ सेंटीमीटर है।
किसकी ऊँचाई सबसे अधिक है?
तर्क: सुधा = १८९, सचिन = १३४, तनुजा = १२२
उत्तर: सुधा
```
*attribute* `height` - *phrasing* `plain`/`genitive`

## `val4_rank` - 4 entities, values premises, 0 transitive hops

```
प्रश्न: ताला की लम्बाई १७३ सेंटीमीटर है। कुरता की लम्बाई १३१ सेंटीमीटर है। साइकिल की लम्बाई २४ सेंटीमीटर है। बाल्टी की लम्बाई १२ सेंटीमीटर है।
लम्बाई के घटते क्रम में दूसरे स्थान पर कौन है?
तर्क: ताला = १७३, कुरता = १३१, साइकिल = २४, बाल्टी = १२
उत्तर: कुरता
```
*attribute* `length` - *phrasing* `plain`/`rank`

```
प्रश्न: संदूक की कीमत ७४० रुपये है। बाल्टी की कीमत १७५ रुपये है। साइकिल की कीमत ३८० रुपये है। ताला की कीमत ४२५ रुपये है।
कीमत के घटते क्रम में दूसरे स्थान पर कौन है?
तर्क: संदूक = ७४०, ताला = ४२५, साइकिल = ३८०, बाल्टी = १७५
उत्तर: ताला
```
*attribute* `price` - *phrasing* `plain`/`rank`

## `eq3_relation` - 3 entities, equality premises, 1 transitive hop

```
प्रश्न: स्वाति, अरुण से लम्बी है। अरुण की ऊँचाई प्रवीण के बराबर है।
स्वाति की ऊँचाई प्रवीण से अधिक है, कम है या बराबर?
तर्क: स्वाति > अरुण, अरुण = प्रवीण
उत्तर: अधिक
```
*attribute* `height` - *phrasing* `adjectival`/`relation`

```
प्रश्न: बबीता के अंक तरुण से अधिक हैं। तरुण के अंक कल्पना से अधिक हैं।
कल्पना के अंक बबीता से अधिक हैं, कम हैं या बराबर?
तर्क: बबीता > तरुण, तरुण > कल्पना
उत्तर: कम
```
*attribute* `marks` - *phrasing* `plain`/`relation`

## `rel4_extreme` - 4 entities, relative premises, 2 transitive hops

```
प्रश्न: बाल्टी की लम्बाई संदूक से ज़्यादा है। साइकिल की लम्बाई ताला से ज़्यादा है। ताला की लम्बाई बाल्टी से ज़्यादा है।
सबसे छोटा कौन है?
तर्क: साइकिल > ताला, ताला > बाल्टी, बाल्टी > संदूक
उत्तर: संदूक
```
*attribute* `length` - *phrasing* `alt_lexical`/`adjectival` - **held-out phrasing**

```
प्रश्न: पंखा की कीमत दरी से अधिक है। साइकिल की कीमत ताला से अधिक है। ताला की कीमत पंखा से अधिक है।
किसकी कीमत सबसे अधिक है?
तर्क: साइकिल > ताला, ताला > पंखा, पंखा > दरी
उत्तर: साइकिल
```
*attribute* `price` - *phrasing* `plain`/`genitive`

## `rel4_pairwise` - 4 entities, relative premises, 2 transitive hops

```
प्रश्न: पंखा की कीमत झाड़ू से ज़्यादा है। झाड़ू की कीमत दरी से ज़्यादा है। ताला की कीमत पंखा से ज़्यादा है।
दरी की कीमत ताला से अधिक है, कम है या बराबर?
तर्क: ताला > पंखा, पंखा > झाड़ू, झाड़ू > दरी
उत्तर: कम
```
*attribute* `price` - *phrasing* `alt_lexical`/`relation` - **held-out phrasing**

```
प्रश्न: कुरता की कीमत साइकिल से अधिक है। साइकिल की कीमत ताला से अधिक है। ताला की कीमत संदूक से अधिक है।
कुरता की कीमत संदूक से अधिक है, कम है या बराबर?
तर्क: कुरता > साइकिल, साइकिल > ताला, ताला > संदूक
उत्तर: अधिक
```
*attribute* `price` - *phrasing* `plain`/`relation`

