;;;; microgpt.lisp

(in-package #:microgpt)

(defun load-names ()
  (let ((path (asdf:system-relative-pathname :microgpt "input.txt"))
        (names (make-array 0 :element-type 'string
                             :adjustable t
                             :fill-pointer 0)))
    (with-open-file (stream path :direction :input)
      (loop for line = (read-line stream nil nil)
            while line
            do (vector-push-extend line names)))
    names))

(defun shuffle (vec)
  "Shuffle vector VEC in place using Fisher-Yates."
  (loop for i from (1- (length vec)) downto 1
        do (rotatef (aref vec i) (aref vec (random (1+ i)))))
  vec)

(defun unique-chars (names)
  "Return a vector of unique characters found across all strings in NAMES."
  (let ((seen (make-hash-table)))
    (loop for name across names
          do (loop for ch across name
                   do (setf (gethash ch seen) t)))
    (loop for ch being the hash-keys of seen
          collect ch into chars
          finally (return (coerce chars 'vector)))))

(defclass value ()
  ((data :initarg :data :accessor data
         :documentation "Scalar value calculated during forward pass.")
   (grad :initform 0 :accessor grad
          :documentation "Derivative of the loss w.r.t. this node.")
   (children :initarg :children :initform '() :accessor children
             :documentation "Children of this node in the computation graph.")
   (local-grads :initarg :local-grads :initform '() :accessor local-grads
                :documentation "Local derivatives w.r.t. children.")))

(defun make-value (data &key (children '()) (local-grads '()))
  (make-instance 'value :data data :children children :local-grads local-grads))

(defun ensure-value (x)
  (if (typep x 'value) x (make-value x)))

(defgeneric value-add (a b)
  (:method ((a value) (b value))
    (make-value (+ (data a) (data b))
                :children (list a b)
                :local-grads (list 1 1)))
  (:method ((a value) b) (value-add a (ensure-value b)))
  (:method (a (b value)) (value-add (ensure-value a) b)))

(defgeneric value-mul (a b)
  (:method ((a value) (b value))
    (make-value (* (data a) (data b))
                :children (list a b)
                :local-grads (list (data b) (data a))))
  (:method ((a value) b) (value-mul a (ensure-value b)))
  (:method (a (b value)) (value-mul (ensure-value a) b)))

(defun value-pow (a n)
  (make-value (expt (data a) n)
              :children (list a)
              :local-grads (list (* n (expt (data a) (1- n))))))

(defun value-log (a)
  (make-value (log (data a))
              :children (list a)
              :local-grads (list (/ 1 (data a)))))

(defun value-exp (a)
  (make-value (exp (data a))
              :children (list a)
              :local-grads (list (exp (data a)))))

(defun value-relu (a)
  (make-value (max 0 (data a))
              :children (list a)
              :local-grads (list (if (> (data a) 0) 1.0 0.0))))

(defun value-neg (a) (value-mul a (make-value -1)))
(defun value-sub (a b) (value-add a (value-neg (ensure-value b))))
(defun value-div (a b) (value-mul a (value-pow (ensure-value b) -1)))

(defun backward (root)
  "Backpropagate gradients from ROOT through the computation graph."
  (let ((topo '())
        (visited (make-hash-table :test 'eq)))
    (labels ((build-topo (v)
               (unless (gethash v visited)
                 (setf (gethash v visited) t)
                 (dolist (child (children v))
                   (build-topo child))
                 (push v topo))))
      (build-topo root))
    (setf (grad root) 1)
    (dolist (v topo)
      (loop for child in (children v)
            for local-grad in (local-grads v)
            do (incf (grad child) (* local-grad (grad v)))))))

;; Box-Muller transform for standard normal (mean 0, std dev 0.08)
(defun random-gauss (&optional (mean 0.0) (std-dev 0.8))
  (let ((u1 (random 1.0))
        (u2 (random 1.0)))
    (+ mean
       (* std-dev
          (sqrt (* -2.0 (log u1)))
          (cos (* 2.0 pi u2))))))

(defun rmsnorm (x)
  (let* ((ms (value-div (reduce #'value-add (mapcar (lambda (xi) (value-mul xi xi)) x))
                        (length x)))
         (scale (value-pow (value-add ms (make-value 1e-5)) -0.5)))
    (mapcar (lambda (xi) (value-mul xi scale)) x)))

(defun softmax (logits)
  (let* ((max-val (reduce #'max logits :key #'data))
         (exps (mapcar (lambda (v) (value-exp (value-sub v (make-value max-val)))) logits))
         (total (reduce #'value-add exps)))
    (mapcar (lambda (e) (value-div e total)) exps)))

(defun linear (x w)
  "Matrix-vector product: each row of W dotted with X."
  (loop for wo in w
        collect (reduce #'value-add (mapcar #'value-mul wo x))))

(defun matrix (nout nin)
  "Create a matrix of value objects with random gaussian init.

Args:
  NOUT (integer): Number of output rows.
  NIN (integer): Number of input columns."
  (loop for _i from 0 below nout
        collect (loop for _j from 0 below nin
                      collect (make-value (random-gauss)))))

(defun gpt (token-id pos-id keys values state-table n-layer n-head head-dim)
  (let* ((tok-emb (nth token-id (gethash "wte" state-table)))
         (pos-emb (nth pos-id (gethash "wpe" state-table)))
         (x (mapcar #'value-add tok-emb pos-emb))
         (x (rmsnorm x)))
    (dotimes (li n-layer)
      (let* ((x-residual x)
             (x-norm (rmsnorm x))
             (q (linear x-norm (gethash (format nil "layer~a.attn_wq" li) state-table)))
             (k (linear x-norm (gethash (format nil "layer~a.attn_wk" li) state-table)))
             (v (linear x-norm (gethash (format nil "layer~a.attn_wv" li) state-table))))
        (push k (aref keys li))
        (push v (aref values li))
        ;; Reverse because we push (append equivalent)
        (let ((cached-keys (reverse (aref keys li)))
              (cached-values (reverse (aref values li)))
              (x-attn '()))
          (dotimes (h n-head)
            (let* ((hs (* h head-dim))
                   (q-h (subseq q hs (+ hs head-dim)))
                   (k-h (mapcar (lambda (ki) (subseq ki hs (+ hs head-dim))) cached-keys))
                   (v-h (mapcar (lambda (vi) (subseq vi hs (+ hs head-dim))) cached-values))
                   (scale (make-value (expt head-dim 0.5)))
                   (attn-logits (loop for kt in k-h
                                      collect (value-div
                                               (reduce #'value-add (mapcar #'value-mul q-h kt))
                                               scale)))
                   (attn-weights (softmax attn-logits))
                   (head-out (loop for j from 0 below head-dim
                                   collect (reduce #'value-add
                                                   (loop for tt from 0 below (length v-h)
                                                         collect (value-mul (nth tt attn-weights)
                                                                            (nth j (nth tt v-h))))))))
              (setf x-attn (append x-attn head-out))))
          (setf x (linear x-attn (gethash (format nil "layer~a.attn_wo" li) state-table)))
          (setf x (mapcar #'value-add x x-residual))
          ;; MLP block
          (let ((x-residual2 x))
            (setf x (rmsnorm x))
            (setf x (linear x (gethash (format nil "layer~a.mlp_fc1" li) state-table)))
            (setf x (mapcar #'value-relu x))
            (setf x (linear x (gethash (format nil "layer~a.mlp_fc2" li) state-table)))
            (setf x (mapcar #'value-add x x-residual2))))))
    (linear x (gethash "lm_head" state-table))))

(defun main ()
  (let* ((names (let ((n (shuffle (load-names))))
                  (format t "Names: ~a~%" (length n))
                  n))
         ;; Let there be a Tokenizer
         ;; to translate strings to discrete symbols and back
         ;;
         ;; unique characters in the dataset become token ids 0..n-1
         (uchars (sort (unique-chars names) #'char<))
         ;; token id for the special Beginning of Sequence (BOS) token
         (bos (length uchars))
         ;; total number of unique tokens, +1 is for BOS
         (vocab-size (let ((v (+ bos 1)))
                       (format t "Vocab size: ~a~%" v)
                       v))
         (n-embd 16)  ;; embedding dimension
         (n-head 4)   ;; number of attention heads
         (n-layer 1)  ;; number of layers
         (block-size 16) ;; maximum sequence length
         (head-dim (/ n-embd n-head)) ;; dimension of each head
         (state-table (let ((ht (make-hash-table :test 'equal)))
                        (setf (gethash "wte" ht) (matrix vocab-size n-embd)
                              (gethash "wpe" ht) (matrix block-size n-embd)
                              (gethash "lm_head" ht) (matrix vocab-size n-embd))
                        (loop for i from 0 below n-layer
                              do (setf (gethash (format nil "layer~a.attn_wq" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.attn_wk" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.attn_wv" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.attn_wo" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.mlp_fc1" i) ht) (matrix (* 4 n-embd) n-embd)
                                       (gethash (format nil "layer~a.mlp_fc2" i) ht) (matrix n-embd (* 4 n-embd))))
                        ht))
         ;; Flatten params into a single vector
         (params (let ((acc (make-array 0 :adjustable t :fill-pointer 0)))
                   (maphash (lambda (k mat)
                              (declare (ignore k))
                              (dolist (row mat)
                                (dolist (p row)
                                  (vector-push-extend p acc))))
                            state-table)
                   acc))
         (num-params (length params))
         (learning-rate 0.01)
         (beta1 0.85)
         (beta2 0.99)
         (eps-adam 1e-8)
         (m (make-array num-params :initial-element 0.0))
         (v (make-array num-params :initial-element 0.0))
         (num-steps 1000) ;; number of training steps
         )
    (format t "Num params: ~a~%" num-params)
    (dotimes (step num-steps)
      ;; Take single document, tokenize it, surround with BOS on both sides
      (let* ((doc (aref names (mod step (length names))))
             (tokens (concatenate 'vector
                                  (vector bos)
                                  (map 'vector (lambda (ch) (position ch uchars)) doc)
                                  (vector bos)))
             (n (min block-size (1- (length tokens))))
             ;; Fresh KV cache per document
             (keys (make-array n-layer :initial-element '()))
             (vals (make-array n-layer :initial-element '()))
             (losses '()))
        ;; Forward pass: build computation graph through to the loss
        (dotimes (pos-id n)
          (let* ((token-id (aref tokens pos-id))
                 (target-id (aref tokens (1+ pos-id)))
                 (logits (gpt token-id pos-id keys vals state-table n-layer n-head head-dim))
                 (probs (softmax logits))
                 (loss-t (value-neg (value-log (nth target-id probs)))))
            (push loss-t losses)))
        (let ((loss (value-mul (make-value (/ 1.0 n)) (reduce #'value-add losses))))
          ;; Backward pass
          (backward loss)
          ;; Adam optimizer update
          (let ((lr-t (* learning-rate (- 1 (/ step num-steps)))))
            (loop for i from 0 below num-params
                  for p = (aref params i)
                  do (setf (aref m i) (+ (* beta1 (aref m i))
                                         (* (- 1 beta1) (grad p))))
                     (setf (aref v i) (+ (* beta2 (aref v i))
                                         (* (- 1 beta2) (expt (grad p) 2))))
                     (let ((m-hat (/ (aref m i) (- 1 (expt beta1 (1+ step)))))
                           (v-hat (/ (aref v i) (- 1 (expt beta2 (1+ step))))))
                       (decf (data p) (* lr-t (/ m-hat (+ (expt v-hat 0.5) eps-adam)))))
                     (setf (grad p) 0)))
          (format t "step ~4d / ~4d | loss ~,4f~%" (1+ step) num-steps (data loss)))))
    ;; Inference: may the model babble back to us
    (let ((temperature 0.5))
      (format t "~%--- inference (new, hallucinated names) ---~%")
      (dotimes (sample-idx 20)
        (let ((keys (make-array n-layer :initial-element '()))
              (vals (make-array n-layer :initial-element '()))
              (token-id bos)
              (sample '()))
          (dotimes (pos-id block-size)
            (let* ((logits (gpt token-id pos-id keys vals state-table n-layer n-head head-dim))
                   (scaled (mapcar (lambda (l) (value-div l (make-value temperature))) logits))
                   (probs (softmax scaled))
                   (weights (mapcar #'data probs))
                   (r (* (random 1.0) (reduce #'+ weights)))
                   (chosen (loop for w in weights
                                 for idx from 0
                                 summing w into cumul
                                 when (>= cumul r) return idx)))
              (setf token-id chosen)
              (when (= token-id bos) (return))
              (push (aref uchars token-id) sample)))
          (format t "sample ~2d: ~{~a~}~%" (1+ sample-idx) (reverse sample)))))))
