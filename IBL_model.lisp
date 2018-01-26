(clear-all)
(require-extra "blending")

(define-model StarCraft-IBL
;; Edit these. These are parameters used in swarm model.
(sgp :esc t    ;;set to t to use subsymbolic computations
     :sim-hook cosine-similarity ;;need to edit the linear-similarity function or just use default?
     :bln t    ;;added this to ensure blending was enabled, but not sure if it is actually doing anything...
     ;;some parameters moved from +actr-parameters+ in swarm model
     :ans 0.25 ;;activation noise, default is nil, swarm was .75 (M-turkers), Christian recommended .25 as a start.
     :tmp nil  ;;decouple noise if not nil, swarm was .7, perhaps change later.
     :mp 2.5   ;;partial matching enabled, default is off/nil, start high (2.5 is really high) and move to 1.
     :bll 0.5  ;;base-level learning enabled, default is off, this is the recommended value.
     :rt -10.0 ;;retrieval threshold, default is 0, think about editing this number.
     :blc 5    ;;base-level learning constant, default is 0, think about editing this number.
     :lf 0.25  ;;latency factor for retrieval times, default is 1.0, think about editing this number.
     :ol nil   ;;affects which base-level learning equation is used, default is nil; use 1 later maybe
     :md -2.5  ;;maximum difference, default similarity value between chunks, default is 1.0, maybe edit this.
     :ga 0.0   ;;set goal activation to 0.0 for spreading activation
     :imaginal-activation 0.0 ;;set imaginal activation to 0.0 for spreading activation
     ;;back to some parameters in the original sgp here
     :v t      ;;set to nil to turn off model output
     :blt t    ;;set to nil to turn off blending trace
     :trace-detail high ;;lower this as needed, start at high for initial debugging.
     :style-warnings t  ;;set to nil to turn off production warnings, start at t for initial debugging.
     ) ;;end sgp
(P 