
### Hi Mau! try with thi function and let me know. I did not work on it enough I think. I can work on that during weekend




import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import multiprocessing

def Determine_Blinking_Distribution5(LocalizationsFinal, Frame_Information, Pre_A, Resolution):
    """
    Traduzione fedele della funzione MATLAB Determine_Blinking_Distribution5 in Python.
    Restituisce:
      bins, D_Counts3, Total_No_Blink, Resolution, X_overall, M_mat
    Input:
      - LocalizationsFinal: list of (N_i x 3) numpy arrays (localizations per image)
      - Frame_Information: list of 1D arrays (frame or time info per localization)
      - Pre_A: integer (max frame difference)
      - Resolution: float (bin width)
    """

    # --- inizializzazioni ---
    X_overall = []

    # 1) determinare il massimo (D_maxf) e minimo (D_maxm) delle distanze (su tutte le immagini)
    D_maxf = 0.0
    D_maxm = np.inf
    for i in range(len(LocalizationsFinal)):
        # calcola distanze pairwise (condensed) come in MATLAB pdist
        D = pdist(LocalizationsFinal[i])
        if D.size == 0:
            continue
        D_max = np.max(D)
        if D_max > D_maxf:
            D_maxf = D_max
        if D_max < D_maxm:
            D_maxm = D_max

    # 2) costruzione dei bins (edge array)
    # MATLAB: bins = [0:Resolution:D_maxf, Inf]
    # Per riprodurre fedelmente includiamo D_maxf come bordo finale (separatore) e poi Inf
#    if D_maxf <= 0:
        # fallback: almeno un bin
#        bin_edges = np.array([0.0, np.inf])
#    else:
#        # generiamo gli edge: 0, Resolution, ..., valore <= D_maxf, poi aggiungiamo D_maxf se non è esatto
#        edges = np.arange(0.0, D_maxf + Resolution, Resolution)
#        if edges[-1] < D_maxf - 1e-12:
#            edges = np.concatenate((edges, [D_maxf]))
#        # inf
#        bin_edges = np.concatenate((edges, [np.inf]))
#    bins = bin_edges

    bins = np.append(np.arange(0, D_maxf, Resolution), np.inf)


    Total_Blink = []
    Total_No_Blink = []

    print('Working on step 1')

    # 3) Per ogni immagine calcolo delle distribuzioni D_Blink e D_No_Blink
    for i in range(len(LocalizationsFinal)):
        # calcola Z2 = pdist([[zeros], Frame_Information[i]])
        frames = np.asarray(Frame_Information[i]).ravel()
        if frames.size == 0:
            # no frames -> skip
            Total_Blink.append(np.zeros(len(bins)-1))
            Total_No_Blink.append(np.zeros(len(bins)-1))
            continue

        frame_zeros = np.zeros(frames.shape)
        frame_data = np.column_stack((frame_zeros, frames))
        Z2 = pdist(frame_data)

        D = pdist(LocalizationsFinal[i])

        # D_Blink: Z2 < Pre_A
        D_Blink = D[Z2 < Pre_A] if D.size > 0 else np.array([])

        # D_No_Blink: (Z2 > Pre_A) & (Z2 < Pre_A*5)
        mask_no_blink = (Z2 > Pre_A) & (Z2 < Pre_A * 5)
        D_No_Blink = D[mask_no_blink] if D.size > 0 else np.array([])

        # histcounts with 'Normalization','prob' -> bin counts divided by total counts (sum=1)
        counts_blink, _ = np.histogram(D_Blink, bins=bins, density=False)
        D_Counts = counts_blink.astype(float)
        if D_Counts.sum() > 0:
            D_Counts = D_Counts / D_Counts.sum()
        # else rimane zero-vector
        Total_Blink.append(D_Counts)

        counts_noblink, _ = np.histogram(D_No_Blink, bins=bins, density=False)
        D_Counts2 = counts_noblink.astype(float)
        if D_Counts2.sum() > 0:
            D_Counts2 = D_Counts2 / D_Counts2.sum()
        Total_No_Blink.append(D_Counts2)

    # Convert lists to numpy arrays (n_images x n_bins)
    Total_Blink = np.vstack([row if row.size else np.zeros(len(bins)-1) for row in Total_Blink])
    Total_No_Blink = np.vstack([row if row.size else np.zeros(len(bins)-1) for row in Total_No_Blink])

    # 4) media delle distribuzioni
    D_Counts = np.mean(Total_Blink, axis=0)
    D_Counts2 = np.mean(Total_No_Blink, axis=0)

    # 5) scala come in MATLAB:
    # D_Scale = sum(D_Counts(10:end)) / sum(D_Counts2(10:end));
    # Nota: MATLAB index 10 -> Python index 9
    # Protezione: se denominatore=0, metti scale = 0
    denom = np.sum(D_Counts2[9:]) if D_Counts2.size >= 9 else 0.0
    numer = np.sum(D_Counts[9:]) if D_Counts.size >= 9 else 0.0
    if denom == 0:
        D_Scale_val = 0.0
    else:
        D_Scale_val = numer / denom

    D_Counts3 = D_Counts - D_Counts2 * D_Scale_val

    # normalizza come in MATLAB: D_Counts3 = (D_Counts3)/sum(D_Counts3);
    sumD3 = np.sum(D_Counts3)
    if sumD3 != 0:
        D_Counts3 = D_Counts3 / sumD3
    else:
        # se tutto zero, mantieni zero vector
        D_Counts3 = D_Counts3.copy()

    # 6) pulizia monotonia (copia fedele della logica MATLAB)
    good = True
    ins = 0
    # MATLAB: for i=4:length(D_Counts3)-1  -> Python indices 3 .. len-2 inclusive
    for idx in range(3, len(D_Counts3) - 1):
        if D_Counts3[idx] > 0 and D_Counts3[idx+1] < D_Counts3[idx] and good:
            # keep going
            continue
        else:
            if ins == 0:
                ins = idx
            good = False
            D_Counts3[idx] = 0.0

    # clamp negatives then normalize twice as MATLAB
    D_Counts3[D_Counts3 < 0] = 0.0
    s = D_Counts3.sum()
    if s != 0:
        D_Counts3 = D_Counts3 / s
    s2 = D_Counts3.sum()
    if s2 != 0:
        D_Counts3 = D_Counts3 / s2

    # noise elimination: MATLAB uses D_Counts3(8:end) -> python index 7:
    if np.sum(D_Counts3[7:] > 0) > 1:
        print('Warning: Eliminating Noise for higher bins')
        D_Counts3[7:] = 0.0
        s3 = D_Counts3.sum()
        if s3 != 0:
            D_Counts3 = D_Counts3 / s3

    Distribution_for_Blink2 = D_Counts3.copy()

    # 7) Fitting: calcolo Dscale_store per ogni immagine e ogni w in 1:Pre_A
    n_images = len(LocalizationsFinal)
    Dscale_store = [ [] for _ in range(n_images) ]  # list of lists

    print('Still Working on step 1, wait for me')

    # funzione per fit: risolve y ~ x * T + (1-x) * B  (x scalare)
    def fit_scalar_scale(T, B, y):
        # tstack simile a MATLAB: t = [True_Distribuiton2; Distribution_for_Blink2]
        # model: F(x) = x*T + (1-x)*B
        t = np.vstack((T, B))  # shape (2, n_bins)

        # residual function for least_squares: scalar x
        def residual(x):
            return (x[0] * t[0, :] + (1.0 - x[0]) * t[1, :]) - y

        # iniziale x0 = 1 (come MATLAB)
        x0 = np.array([1.0])

        # use 'lm' (unconstrained) to mimic lsqcurvefit w/o bounds
        # if y or t are all zeros, return 0 to avoid problems
        if np.all(y == 0):
            return 0.0

        try:
            res = least_squares(residual, x0, method='lm', xtol=1e-6, ftol=1e-6, gtol=1e-6)
            x_est = float(res.x[0])
        except Exception:
            # fallback robust calculation (linear regression scalar)
            A = (t[0, :] - t[1, :])
            denom = np.sum(A * A)
            if denom == 0:
                x_est = 0.0
            else:
                x_est = np.sum(A * (y - t[1, :])) / denom
        return x_est

    # worker per immagine (seriale o parallelo)
    def process_image_fit(i):
        frames = np.asarray(Frame_Information[i]).ravel()
        if frames.size == 0:
            return [0.0] * Pre_A

        frame_zeros = np.zeros(frames.shape)
        frame_data = np.column_stack((frame_zeros, frames))
        Z2 = pdist(frame_data)

        D = pdist(LocalizationsFinal[i])

        # D_No_Blink as in MATLAB
        mask_no_blink = (Z2 > Pre_A) & (Z2 < Pre_A * 5)
        D_No_Blink = D[mask_no_blink] if D.size > 0 else np.array([])

        counts_nb, _ = np.histogram(D_No_Blink, bins=bins, density=False)
        True_Distribuiton2 = counts_nb.astype(float)
        if True_Distribuiton2.sum() > 0:
            True_Distribuiton2 = True_Distribuiton2 / True_Distribuiton2.sum()

        # per ogni lag w
        dscale_res = []
        for w in range(1, Pre_A + 1):
            # D_Blink where Z2 == w
            if D.size > 0:
                mask_blink = (Z2 == w)
                D_Blink = D[mask_blink]
            else:
                D_Blink = np.array([])

            counts_blink, _ = np.histogram(D_Blink, bins=bins, density=False)
            Temp_Distribution = counts_blink.astype(float)
            if Temp_Distribution.sum() > 0:
                Temp_Distribution = Temp_Distribution / Temp_Distribution.sum()
            else:
                Temp_Distribution = Temp_Distribution  # zeros

            # fit x s.t. y ~ x*T + (1-x)*B  where t = [True_Distribuiton2; Distribution_for_Blink2]
            y = Temp_Distribution
            x_est = fit_scalar_scale(True_Distribuiton2, Distribution_for_Blink2, y)

            dscale_res.append(float(x_est))

        return dscale_res

    # parallelo con joblib (o seriale se fallisce)
    try:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=min(num_cores, n_images))(
            delayed(process_image_fit)(i) for i in range(n_images)
        )
        for i, res in enumerate(results):
            Dscale_store[i] = res
    except Exception as e:
        print("Parallel processing failed, falling back to serial processing:", e)
        for i in range(n_images):
            Dscale_store[i] = process_image_fit(i)

    # 8) ricombina Dscale_store in matrice (n_images x Pre_A), come in MATLAB Dscale_store2
    # attenzione: se qualche immagine ha meno elementi, riempiamo con zeri
    Dscale_store2 = np.zeros((n_images, Pre_A), dtype=float)
    for i in range(n_images):
        arr = np.array(Dscale_store[i], dtype=float)
        if arr.size < Pre_A:
            # pad con zeri
            padded = np.zeros(Pre_A, dtype=float)
            padded[:arr.size] = arr
            arr = padded
        Dscale_store2[i, :] = arr

    # MATLAB: se più immagini -> X_overall = mean(Dscale_store2); Dscale_store = mean(Dscale_store2)
    if n_images > 1:
        X_overall = np.mean(Dscale_store2, axis=0)
        Dscale_store_mean = np.mean(Dscale_store2, axis=0)
    else:
        X_overall = Dscale_store2.flatten().copy()
        Dscale_store_mean = Dscale_store2.flatten().copy()

    # 9) clamp come MATLAB
    Dscale_store_mean = np.minimum(Dscale_store_mean, 1.0)       # >1 -> 1
    Dscale_store_mean = np.maximum(Dscale_store_mean, 1e-7)     # <0 -> 1e-7
    if Dscale_store_mean.size >= 1:
        Dscale_store_mean[-1] = 1.0                             # ultimo elemento = 1

    # 10) calcolo Deviation_in_Probabilityt per immagine (stessa logica MATLAB)
    Deviation_in_Probabilityt = [None] * n_images

    print('Still going, Working on step 1')
    # worker per M matrix (usa Dscale_store_mean, Distribution_for_Blink2)
    def process_image_m_matrix(i):
        frames = np.asarray(Frame_Information[i]).ravel()
        if frames.size == 0:
            # return zeros matrix shape (Pre_A x n_bins)
            return [np.zeros(len(bins)-1) for _ in range(Pre_A)]

        frame_zeros = np.zeros(frames.shape)
        frame_data = np.column_stack((frame_zeros, frames))
        Z2 = pdist(frame_data)

        D = pdist(LocalizationsFinal[i])

        mask_no_blink = (Z2 > Pre_A) & (Z2 < Pre_A * 5)
        D_No_Blink = D[mask_no_blink] if D.size > 0 else np.array([])

        counts_nb, _ = np.histogram(D_No_Blink, bins=bins, density=False)
        True_Distribuiton = counts_nb.astype(float)
        if True_Distribuiton.sum() > 0:
            True_Distribuiton = True_Distribuiton / True_Distribuiton.sum()

        deviation_results = []
        for w in range(1, Pre_A + 1):
            D_Scale = float(Dscale_store_mean[w-1])  # scalar
            Temp_Distribution2 = Distribution_for_Blink2 * (1.0 - D_Scale) + True_Distribuiton * D_Scale

            # Combined = (Temp_Distribution2 - D_Scale * True_Distribuiton) ./ Temp_Distribution2
            numerator = Temp_Distribution2 - D_Scale * True_Distribuiton
            with np.errstate(divide='ignore', invalid='ignore'):
                Combined = np.divide(numerator, Temp_Distribution2, out=np.zeros_like(numerator), where=Temp_Distribution2 != 0)
                Combined[~np.isfinite(Combined)] = 0.0

            deviation_results.append(Combined)

        return deviation_results

    # parallelo per M matrix
    try:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=min(num_cores, n_images))(
            delayed(process_image_m_matrix)(i) for i in range(n_images)
        )
        for i, res in enumerate(results):
            Deviation_in_Probabilityt[i] = res
    except Exception as e:
        print("Parallel processing for M failed, falling back to serial:", e)
        for i in range(n_images):
            Deviation_in_Probabilityt[i] = process_image_m_matrix(i)

    # 11) calcolo M_mat come media su immagini (per ogni w)
    if n_images > 1:
        M_mat = np.zeros((Pre_A, len(bins)-1), dtype=float)
        for w in range(Pre_A):
            # per immagine i prendo Deviation_in_Probabilityt[i][w]
            M_mat_t = np.array([Deviation_in_Probabilityt[i][w] for i in range(n_images)], dtype=float)
            M_mat[w, :] = np.mean(M_mat_t, axis=0)
    else:
        # se una sola immagine, restituisco la matrice per quella immagine
        M_mat = np.array(Deviation_in_Probabilityt[0])

    # Total_No_Blink: restituiamo la media? In MATLAB Total_No_Blink variable veniva popolata in STEP 1
    # Qui restituiamo Total_No_Blink come media sulle immagini (compatibile con earlier usage)
    Total_No_Blink = Total_No_Blink  # è la matrice (n_images x n_bins) creata prima

    # return: bins, D_Counts3, Total_No_Blink, Resolution, X_overall, M_mat
    return bins, D_Counts3, Total_No_Blink, Resolution, X_overall, M_mat
