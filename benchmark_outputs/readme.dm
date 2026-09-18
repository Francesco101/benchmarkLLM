Per ora solo dei test a vuoto, per verificare che funzioni tutto.

Dentro a multitest/ :
  -> alcuni task non sono andati per dei typos nel nome ( >>lm_eval ls tasks )
  -> ho separato solo tra chat-tests e logits-tests ma va fatto più ad hoc per singola task
   \   -> logits-tests aveva "gen_kwargs" = {} --> vuoto
    \_ -> chat-tests aveva "max_tokens" = 8192  (Quasi sicuramente non basta per task tipo aime24)
  
  -> gemma-[...]-it probabilmente va passato in modalità chat con template anche per far girare test-logits.

  -> in generale per ora:
    \  -> i test logits son pericolosi silenziosamente. (andranno controllati bene)
     \ -> i test chat possono finire i token in reasoning e dare risposte vuote (a volte per solo una frazione del test) falsandolo silenziosamente.
  


