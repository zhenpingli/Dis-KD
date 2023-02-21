# Subword Language Model for Query Auto-Completion

This is the official github repository for [Subword Language Model for Query Auto-Completion](https://arxiv.org/abs/1909.00599) (EMNLP-IJCNLP 2019).


## Dependencies
- Python 3
- PyTorch
- SentencePiece


## Preparing Data
- Dowload original AOL query log dataset: `./get_data.sh`.
  This files will be saved in `data/aol/org` directory.
- Split this data into `{train, valid, test}.{query, uid, time}.txt` by giving name tag for the split and specifying time interval of each split.
  It will be generated in the `data/aol/<tag>` directory.
  Or, you can just run `split.sh` to use a pre-determined partition setting. 
  ```
  python split.py --tag full  --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                              --valid_start "2006-05-18 00:00:00" --valid_end "2006-05-25 00:00:00" \
                              --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00"
  ```
- Train [SentencePiece](https://github.com/google/sentencepiece/) models (char, bpe, and unigram): `./train_spms.sh`. 
  You may change the subword vocabulary size (default: 256).


## Training a language model
```
python train.py \
    --data_dir data/aol/full \
    --spm <spm> \               # char, bpe/<vocab-size>, or unigram/<vocab-size> 
    --sample -1 0.2 \           # if spm is ungiram
    --ninp 100 \
    --nhid 600 \
    --nlayers 1 \
    --max_seq_len 40
```  

## Generating completions using a trained language model
``` 
python generate.py \
    --gen_bsz 1 \
    --beam_size 30 \
    --branching_factor 30 \
    --retrace <R> \             # for the retrace algorithm
    --nbest <n> \               # for the n-best decoding
    --do_merge \                # for marginalization
```


## Citation
If you find this work useful, please cite:
```
@article{kim2019subword,
  title={Subword Language Model for Query Auto-Completion},
  author={Kim, Gyuwan},
  journal={arXiv preprint arXiv:1909.00599},
  year={2019}
}
```


## Contact Information
Please feel free to contact Gyuwan Kim ([gyuwan.kim@navercorp.com](mailto:gyuwan.kim@navercorp.com)) if there is any question.


## License
MIT License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

## Result

## 原文
``` 
作者的LSTM  中间对hidden 做了一堆操作大概就是 各种 sum

没有length
08/31/2021 20:57:48 -  Loading tokenizer from spm/unigram/spm.model
08/31/2021 20:57:48 -    Vocab size: 256
08/31/2021 20:57:48 -  Reading a dataset (all quries from test.query.txt)
08/31/2021 20:57:48 -    Number of test data:     1990 (seen 433, unseen 1557)
08/31/2021 20:57:48 -  Loading model from /home/subword/models/unigram
08/31/2021 20:57:52 -  Generation starts!
08/31/2021 20:59:00 -    mean decode length:  7.1
08/31/2021 20:59:00 -  67.9 s | 34.1 ms/query | 29.3 qps
08/31/2021 20:59:00 -  mrr @ 1: 0.4596 (seen) 0.1265 (unseen) 0.1990 (all)
08/31/2021 20:59:00 -  mrr @ 2: 0.5277 (seen) 0.1468 (unseen) 0.2296 (all)
08/31/2021 20:59:00 -  mrr @ 3: 0.5346 (seen) 0.1523 (unseen) 0.2355 (all)
08/31/2021 20:59:00 -  mrr @ 4: 0.5427 (seen) 0.1565 (unseen) 0.2405 (all)
08/31/2021 20:59:00 -  mrr @ 5: 0.5450 (seen) 0.1578 (unseen) 0.2420 (all)
08/31/2021 20:59:00 -  mrr @ 6: 0.5462 (seen) 0.1586 (unseen) 0.2430 (all)
08/31/2021 20:59:00 -  mrr @ 7: 0.5478 (seen) 0.1593 (unseen) 0.2438 (all)
08/31/2021 20:59:00 -  mrr @ 8: 0.5481 (seen) 0.1599 (unseen) 0.2444 (all)
08/31/2021 20:59:00 -  mrr @ 9: 0.5489 (seen) 0.1605 (unseen) 0.2450 (all)
08/31/2021 20:59:00 -  mrr @10: 0.5494 (seen) 0.1609 (unseen) 0.2454 (all)
08/31/2021 20:59:00 -   
08/31/2021 20:59:00 -  pmrr @ 1: 0.5358 (seen) 0.2704 (unseen) 0.3281 (all)
08/31/2021 20:59:00 -  pmrr @ 2: 0.6028 (seen) 0.3086 (unseen) 0.3726 (all)
08/31/2021 20:59:00 -  pmrr @ 3: 0.6097 (seen) 0.3215 (unseen) 0.3842 (all)
08/31/2021 20:59:00 -  pmrr @ 4: 0.6184 (seen) 0.3282 (unseen) 0.3913 (all)
08/31/2021 20:59:00 -  pmrr @ 5: 0.6202 (seen) 0.3320 (unseen) 0.3947 (all)
08/31/2021 20:59:00 -  pmrr @ 6: 0.6217 (seen) 0.3333 (unseen) 0.3961 (all)
08/31/2021 20:59:00 -  pmrr @ 7: 0.6234 (seen) 0.3345 (unseen) 0.3974 (all)
08/31/2021 20:59:00 -  pmrr @ 8: 0.6237 (seen) 0.3357 (unseen) 0.3983 (all)
08/31/2021 20:59:00 -  pmrr @ 9: 0.6245 (seen) 0.3365 (unseen) 0.3992 (all)
08/31/2021 20:59:00 -  pmrr @10: 0.6256 (seen) 0.3373 (unseen) 0.4000 (all)

生成的时候有length

08/31/2021 21:11:07 -  Loading tokenizer from spm/unigram/spm.model
08/31/2021 21:11:07 -    Vocab size: 256
08/31/2021 21:11:07 -  Reading a dataset (all quries from test.query.txt)
08/31/2021 21:11:07 -    Number of test data:     1990 (seen 433, unseen 1557)
08/31/2021 21:11:07 -  Loading model from /home/subword/models/unigram
08/31/2021 21:11:11 -  Generation starts!
08/31/2021 21:12:14 -    mean decode length:  7.3
08/31/2021 21:12:14 -  62.6 s | 31.4 ms/query | 31.8 qps
08/31/2021 21:12:14 -  mrr @ 1: 0.4573 (seen) 0.1259 (unseen) 0.1980 (all)
08/31/2021 21:12:14 -  mrr @ 2: 0.5323 (seen) 0.1471 (unseen) 0.2309 (all)
08/31/2021 21:12:14 -  mrr @ 3: 0.5393 (seen) 0.1544 (unseen) 0.2381 (all)
08/31/2021 21:12:14 -  mrr @ 4: 0.5462 (seen) 0.1579 (unseen) 0.2424 (all)
08/31/2021 21:12:14 -  mrr @ 5: 0.5480 (seen) 0.1592 (unseen) 0.2438 (all)
08/31/2021 21:12:14 -  mrr @ 6: 0.5496 (seen) 0.1602 (unseen) 0.2450 (all)
08/31/2021 21:12:14 -  mrr @ 7: 0.5506 (seen) 0.1608 (unseen) 0.2456 (all)
08/31/2021 21:12:14 -  mrr @ 8: 0.5517 (seen) 0.1613 (unseen) 0.2462 (all)
08/31/2021 21:12:14 -  mrr @ 9: 0.5527 (seen) 0.1620 (unseen) 0.2470 (all)
08/31/2021 21:12:14 -  mrr @10: 0.5532 (seen) 0.1622 (unseen) 0.2473 (all)
08/31/2021 21:12:14 -   
08/31/2021 21:12:14 -  pmrr @ 1: 0.5312 (seen) 0.2704 (unseen) 0.3271 (all)
08/31/2021 21:12:14 -  pmrr @ 2: 0.6051 (seen) 0.3080 (unseen) 0.3726 (all)
08/31/2021 21:12:14 -  pmrr @ 3: 0.6112 (seen) 0.3232 (unseen) 0.3858 (all)
08/31/2021 21:12:14 -  pmrr @ 4: 0.6187 (seen) 0.3293 (unseen) 0.3923 (all)
08/31/2021 21:12:14 -  pmrr @ 5: 0.6215 (seen) 0.3326 (unseen) 0.3955 (all)
08/31/2021 21:12:14 -  pmrr @ 6: 0.6234 (seen) 0.3344 (unseen) 0.3973 (all)
08/31/2021 21:12:14 -  pmrr @ 7: 0.6244 (seen) 0.3356 (unseen) 0.3985 (all)
08/31/2021 21:12:14 -  pmrr @ 8: 0.6259 (seen) 0.3368 (unseen) 0.3997 (all)
08/31/2021 21:12:14 -  pmrr @ 9: 0.6272 (seen) 0.3380 (unseen) 0.4009 (all)
08/31/2021 21:12:14 -  pmrr @10: 0.6276 (seen) 0.3387 (unseen) 0.4016 (all)


pytorch  的LSTM


08/31/2021 21:44:50 -  Loading tokenizer from spm/unigram/spm.model
08/31/2021 21:44:50 -    Vocab size: 256
08/31/2021 21:44:50 -  Reading a dataset (all quries from test.query.txt)
08/31/2021 21:44:50 -    Number of test data:     1990 (seen 433, unseen 1557)
08/31/2021 21:44:50 -  Loading model from /home/subword/models/unigram
08/31/2021 21:44:54 -  Generation starts!
08/31/2021 21:45:43 -    mean decode length:  7.3
08/31/2021 21:45:43 -  49.4 s | 24.8 ms/query | 40.3 qps
08/31/2021 21:45:43 -  mrr @ 1: 0.4503 (seen) 0.1278 (unseen) 0.1980 (all)
08/31/2021 21:45:43 -  mrr @ 2: 0.5115 (seen) 0.1455 (unseen) 0.2251 (all)
08/31/2021 21:45:43 -  mrr @ 3: 0.5216 (seen) 0.1498 (unseen) 0.2307 (all)
08/31/2021 21:45:43 -  mrr @ 4: 0.5279 (seen) 0.1533 (unseen) 0.2348 (all)
08/31/2021 21:45:43 -  mrr @ 5: 0.5325 (seen) 0.1541 (unseen) 0.2364 (all)
08/31/2021 21:45:43 -  mrr @ 6: 0.5360 (seen) 0.1552 (unseen) 0.2381 (all)
08/31/2021 21:45:43 -  mrr @ 7: 0.5370 (seen) 0.1565 (unseen) 0.2393 (all)
08/31/2021 21:45:43 -  mrr @ 8: 0.5376 (seen) 0.1572 (unseen) 0.2399 (all)
08/31/2021 21:45:43 -  mrr @ 9: 0.5388 (seen) 0.1577 (unseen) 0.2406 (all)
08/31/2021 21:45:43 -  mrr @10: 0.5395 (seen) 0.1580 (unseen) 0.2410 (all)
08/31/2021 21:45:43 -   
08/31/2021 21:45:43 -  pmrr @ 1: 0.5150 (seen) 0.2685 (unseen) 0.3221 (all)
08/31/2021 21:45:43 -  pmrr @ 2: 0.5762 (seen) 0.3019 (unseen) 0.3616 (all)
08/31/2021 21:45:43 -  pmrr @ 3: 0.5908 (seen) 0.3132 (unseen) 0.3736 (all)
08/31/2021 21:45:43 -  pmrr @ 4: 0.5972 (seen) 0.3211 (unseen) 0.3812 (all)
08/31/2021 21:45:43 -  pmrr @ 5: 0.6027 (seen) 0.3234 (unseen) 0.3842 (all)
08/31/2021 21:45:43 -  pmrr @ 6: 0.6062 (seen) 0.3264 (unseen) 0.3873 (all)
08/31/2021 21:45:43 -  pmrr @ 7: 0.6072 (seen) 0.3285 (unseen) 0.3891 (all)
08/31/2021 21:45:43 -  pmrr @ 8: 0.6078 (seen) 0.3299 (unseen) 0.3904 (all)
08/31/2021 21:45:43 -  pmrr @ 9: 0.6090 (seen) 0.3308 (unseen) 0.3913 (all)
08/31/2021 21:45:43 -  pmrr @10: 0.6097 (seen) 0.3314 (unseen) 0.3920 (all)
08/31/2021 21:45:43 -   

Process finished with exit code 0



``` 
## Transformers
``` 
10/14/2021 18:55:21 -  Loading tokenizer from spm/unigram/spm.model
10/14/2021 18:55:21 -    Vocab size: 256
10/14/2021 18:55:21 -  Reading a dataset (all quries from test.query.txt)
10/14/2021 18:55:21 -    Number of test data:     4330 (seen 1064, unseen 3266)
10/14/2021 18:55:21 -  Loading model from /home/qac/models/unigram
10/14/2021 18:55:25 -  Generation starts!
10/14/2021 19:00:39 -    mean decode length:  7.9
10/14/2021 19:00:39 -  314.3 s | 72.6 ms/query | 13.8 qps
10/14/2021 19:00:39 -  mrr @ 1: 0.5019 (seen) 0.1448 (unseen) 0.2326 (all)
10/14/2021 19:00:39 -  mrr @ 2: 0.5555 (seen) 0.1661 (unseen) 0.2618 (all)
10/14/2021 19:00:39 -  mrr @ 3: 0.5752 (seen) 0.1721 (unseen) 0.2712 (all)
10/14/2021 19:00:39 -  mrr @ 4: 0.5811 (seen) 0.1755 (unseen) 0.2752 (all)
10/14/2021 19:00:39 -  mrr @ 5: 0.5856 (seen) 0.1772 (unseen) 0.2776 (all)
10/14/2021 19:00:39 -  mrr @ 6: 0.5882 (seen) 0.1786 (unseen) 0.2792 (all)
10/14/2021 19:00:39 -  mrr @ 7: 0.5894 (seen) 0.1794 (unseen) 0.2801 (all)
10/14/2021 19:00:39 -  mrr @ 8: 0.5899 (seen) 0.1799 (unseen) 0.2807 (all)
10/14/2021 19:00:39 -  mrr @ 9: 0.5906 (seen) 0.1804 (unseen) 0.2812 (all)
10/14/2021 19:00:39 -  mrr @10: 0.5910 (seen) 0.1808 (unseen) 0.2816 (all)
10/14/2021 19:00:39 -   
10/14/2021 19:00:39 -  pmrr @ 1: 0.5451 (seen) 0.2716 (unseen) 0.3388 (all)
10/14/2021 19:00:39 -  pmrr @ 2: 0.6048 (seen) 0.3085 (unseen) 0.3813 (all)
10/14/2021 19:00:39 -  pmrr @ 3: 0.6220 (seen) 0.3219 (unseen) 0.3956 (all)
10/14/2021 19:00:39 -  pmrr @ 4: 0.6281 (seen) 0.3277 (unseen) 0.4016 (all)
10/14/2021 19:00:39 -  pmrr @ 5: 0.6334 (seen) 0.3311 (unseen) 0.4053 (all)
10/14/2021 19:00:39 -  pmrr @ 6: 0.6368 (seen) 0.3346 (unseen) 0.4088 (all)
10/14/2021 19:00:39 -  pmrr @ 7: 0.6379 (seen) 0.3359 (unseen) 0.4101 (all)
10/14/2021 19:00:39 -  pmrr @ 8: 0.6385 (seen) 0.3371 (unseen) 0.4112 (all)
10/14/2021 19:00:39 -  pmrr @ 9: 0.6393 (seen) 0.3381 (unseen) 0.4121 (all)
10/14/2021 19:00:39 -  pmrr @10: 0.6400 (seen) 0.3386 (unseen) 0.4127 (all)



config:
  "n_ctx": 1024,
  "n_embd": 512,
  "n_head": 8,
  "n_layer": 8,
  "n_positions": 1024,

``` 
