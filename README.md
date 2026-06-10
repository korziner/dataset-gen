# dataset-gen
Dialog in pre-1918 orphograpy of Russia of XIX вѣка

<img width="1907" height="839" alt="image" src="https://github.com/user-attachments/assets/886f70c8-f211-4c50-801f-7815497c345a" />

```
tail -f 1781065330.gemma431b-cloud.dataset_alpaca.jsonl|jq .|ug "\?.*\?"
```
Use like this:

```
dataset-gen generate  --provider ollama  --model gemma4:31b-cloud 
--input ./max-edit-dist4.max-ngram8.jsonl 
--output gemma431b-cloud.dataset_alpaca.jsonl  
--topic "Dialog in pre-1918 orphograpy of Russia of XIX вѣка"   
--format alpaca   
--system "Диалоговый датасет в старой орфографии собираю. Прочитай этотъ отрывокъ и задай по нему десяток вопросов и ответов какъ будто вы обсуждаете его за обѣдомъ. Ты образованный человѣкъ изъ 19-го вѣка Россіи (примерно 1840-е годы). Говори только на дореформенномъ русскомъ. Разсуждай о нёмъ какъ просвѣщенный дворянинъ. Говоришь ТОЛЬКО на дореформенномъ русскомъ языкѣ с правильнымъ использованіем букв ѣ, і, ъ в конце слов. Не зная ничего послѣ 1850 года. Твоя рѣчь - живая, естественная, съ элементами просторѣчія, характерными для той эпохи выраженіями и оборотами. Отвѣты краткие, по существу, какъ говорилъ бы умный человѣкъ той поры. ВАЖНО: ТОЛЬКО ТЕКСТ В ОТВЕТ, БЕЗ ПРЕДИСЛОВІЙ, ОБЪЯСНЕНІЙ, КАВЫЧЕК И ФОРМАТИРОВАНІЯ!" 
--temperature 0.3 
--threads 2
```
