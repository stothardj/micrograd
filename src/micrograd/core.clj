(ns micrograd.core
  (:require [clojure.string :as str]
            [clojure.math :as math]))

;; Lol
(defprotocol Valuable
  (to-val [x]))

(defprotocol IValue
  (get-data [this])
  (get-grad [this])
  (inc-grad [this n])
  (backward [this])
  (run-backward [this])
  (set-backward [this b]))

;; grad is mutable so it can be updated for each backward pass.
;; backward is only mutable so it can be set in a way where the function can depend on calling
;; get-grad on the node parent node. Otherwise it would require a (nilable) parameter or a
;; dynamic variable, both of which seem worse.
(deftype Value [data ^:volatile-mutable grad op children ^:volatile-mutable backward]
  Valuable
  (to-val [x] x)
  IValue
  (get-data [this] data)
  (get-grad [this] grad)
  (inc-grad [this n] (set! grad (+ grad n)))
  (backward [this]
    (letfn [(build-topo [v & {:keys [accum visited] :as opts}]
              (if (visited v)
                opts
                (->
                 (reduce #(build-topo %2 %1) opts (.children v))
                 (update :accum #(conj % v)))))]
      (set! grad 1)
      (doseq [v (:accum (build-topo this :accum '() :visited #{}))]
        (run-backward v))))
  (run-backward [this] (backward))
  (set-backward [this b] (set! backward b))
  Object
  (toString [this] (str "Value{data: " data
                        " grad: " grad
                        " op: " op
                        " children: [" (str/join ", " (map str children)) "]}")))

(defn make-value [data & {:keys [grad op children backward]
                          :or {grad 0 op "" children [] backward (fn [])}}]
  (->Value data grad op children backward))

(extend-protocol Valuable
  java.lang.Long
  (to-val [x] (make-value x))
  java.lang.Integer
  (to-val [x] (make-value x))
  java.lang.Double
  (to-val [x] (make-value x))
  Number
  (to-val [x] (make-value x)))

(defn add [x y]
  (let [vx (to-val x)
        vy (to-val y)
        out (make-value (+ (get-data vx) (get-data vy))
                        :op "+" :children [vx vy])
        backward (fn []
                   (let [outg (get-grad out)]
                     (inc-grad vx outg)
                     (inc-grad vy outg)))]
    (set-backward out backward)
    out))

(defn mul [x y]
  (let [vx (to-val x)
        vy (to-val y)
        out (make-value (* (get-data vx) (get-data vy))
                        :op "*" :children [vx vy])
        backward (fn []
                   (let [vxd (get-data vx)
                         vyd (get-data vy)
                         outg (get-grad out)]
                     (inc-grad vx (* vyd outg))
                     (inc-grad vy (* vxd outg))))]
    (set-backward out backward)
    out))

(defn pow [x n]
  (let [vx (to-val x)
        out (make-value (math/pow (get-data vx) n)
                        :op (str "^" n)
                        :children [vx])
        backward (fn []
                   (inc-grad vx (* n (math/pow (get-data vx) (- n 1)) (get-grad out))))]
    (set-backward out backward)
    out))

;; Composite operators
(defn neg [x] (mul x -1))
(defn sub [x y] (add x (neg y)))
(defn div [x y] (mul x (pow y -1)))
