(ns micrograd.nn
  (:require [micrograd.core :as core]))

(defprotocol Module
  (parameters [this]))

(defrecord Neuron [w b nonlin]
  Module
  (parameters [this] (conj w b)))

(defn make-neuron [nin & {:keys [nonlin] :or {nonlin true}}]
  (->Neuron (repeatedly nin #(core/make-value (- (rand 2) 1)))
            (core/make-value 0)
            nonlin))

(defn apply-neuron [n x]
  (let [lin (->> (map core/mul (:w n) x)
             (reduce core/add)
             (core/add (:b n)))]
    (if (:nonlin n) (core/relu lin) lin)))

(defrecord Layer [neurons]
  Module
  (parameters [this] (mapcat parameters neurons)))

(defn make-layer [nin nout & {:keys [nonlin] :as opts}]
  (->Layer (repeatedly nout #(apply make-neuron (conj (mapcat identity opts) nin)))))

(defn apply-layer [l x]
  (map #(apply-neuron % x) (:neurons l)))
