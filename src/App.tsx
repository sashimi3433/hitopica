import React, { useState, useEffect, useRef } from 'react';
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { 
  Upload, 
  Download, 
  RefreshCw, 
  Image as ImageIcon, 
  ArrowLeft,
  Eye,
  Info
} from 'lucide-react';

// お手本用のデフォルト画像（美しいポートレート）
const DEMO_IMAGE_URL = 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&q=80&w=1200';

// 固定のHDR発光パラメータ
const HDR_PARAMS = {
  glowStrength: 1.6,   // 光の強さ
  glowRadius: 45,      // 光の広がり (ピクセル)
  exposure: 1.3,       // 人物の明るさ
  saturation: 1.25,    // 人物の色鮮やかさ
  threshold: 0.18      // 光らせる基準（明るい部分を優先）
};

export default function App() {
  // 画像の状態
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [aspectRatio, setAspectRatio] = useState<number | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);

  // AIモデルの状態
  const [segmenter, setSegmenter] = useState<ImageSegmenter | null>(null);
  const [modelProgress, setModelProgress] = useState(0);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);

  // 処理結果と表示制御
  const [maskCanvas, setMaskCanvas] = useState<HTMLCanvasElement | null>(null);
  const [showOriginal, setShowOriginal] = useState(false); // 元画像表示フラグ

  // 参照
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 1. 初回起動時にAIモデルをローカルからダウンロードしてセットアップ
  useEffect(() => {
    async function initAiModel() {
      try {
        setIsModelLoading(true);
        setModelProgress(10);

        // 同一サーバー（Cloudflare）の /wasm ディレクトリからリゾルバーを読み込む
        const wasmDirectory = window.location.origin + "/wasm";
        const vision = await FilesetResolver.forVisionTasks(wasmDirectory);
        setModelProgress(30);

        // 同一サーバー（Cloudflare）の /models からモデルを読み込む
        const modelUrl = window.location.origin + "/models/selfie_segmenter.tflite";
        const response = await fetch(modelUrl);
        
        if (!response.ok) {
          throw new Error('モデルデータの読み込みに失敗しました。サーバーの設定またはネットワークを確認してください。');
        }

        const reader = response.body?.getReader();
        const contentLength = +(response.headers.get('Content-Length') || 0);

        let receivedLength = 0;
        const chunks: BlobPart[] = [];
        
        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            receivedLength += value.length;
            
            const downloadProgress = contentLength ? (receivedLength / contentLength) : 0.5;
            setModelProgress(Math.floor(30 + downloadProgress * 55));
          }
        }

        const blob = new Blob(chunks);
        const modelBlobUrl = URL.createObjectURL(blob);
        setModelProgress(90);

        const imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: modelBlobUrl,
            delegate: "GPU"
          },
          runningMode: "IMAGE",
          outputCategoryMask: false,
          outputConfidenceMasks: true // なめらかな輪郭抽出
        });

        setSegmenter(imageSegmenter);
        setModelProgress(100);
        
        setTimeout(() => {
          setIsModelLoading(false);
        }, 500);

      } catch (err: any) {
        console.error("AI Model Initialization Error:", err);
        setModelError(err.message || '準備中にエラーが発生しました。ページを再読み込みしてください。');
        setIsModelLoading(false);
      }
    }

    initAiModel();
  }, []);

  // 2. 画像がセットされたら自動でセグメンテーションを実行
  useEffect(() => {
    if (!imageEl || !segmenter) return;

    async function runSegmentation() {
      setIsSegmenting(true);
      try {
        const result = segmenter!.segment(imageEl!);
        const confidenceMasks = result.confidenceMasks;

        if (confidenceMasks && confidenceMasks.length > 0) {
          const mask = confidenceMasks[0];
          const maskData = mask.getAsFloat32Array();
          
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = mask.width;
          tempCanvas.height = mask.height;
          const tempCtx = tempCanvas.getContext('2d');
          
          if (tempCtx) {
            const tempImgData = tempCtx.createImageData(mask.width, mask.height);
            for (let i = 0; i < maskData.length; i++) {
              const confidence = maskData[i]; // 0.0 ~ 1.0
              const idx = i * 4;
              
              tempImgData.data[idx] = 255;
              tempImgData.data[idx+1] = 255;
              tempImgData.data[idx+2] = 255;
              tempImgData.data[idx+3] = Math.round(confidence * 255);
            }
            tempCtx.putImageData(tempImgData, 0, 0);
            
            const finalMaskCanvas = document.createElement('canvas');
            finalMaskCanvas.width = imageEl!.naturalWidth;
            finalMaskCanvas.height = imageEl!.naturalHeight;
            const finalMaskCtx = finalMaskCanvas.getContext('2d');
            if (finalMaskCtx) {
              finalMaskCtx.drawImage(tempCanvas, 0, 0, finalMaskCanvas.width, finalMaskCanvas.height);
              setMaskCanvas(finalMaskCanvas);
            }
          }
        }
      } catch (err) {
        console.error("Segmentation Failed:", err);
      } finally {
        setIsSegmenting(false);
      }
    }

    runSegmentation();
  }, [imageEl, segmenter]);

  // 3. 描画パイプライン（自動でHDR発光効果を適用）
  useEffect(() => {
    if (!imageEl || !canvasRef.current || !maskCanvas) return;

    const canvas = canvasRef.current;
    canvas.width = imageEl.naturalWidth;
    canvas.height = imageEl.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // A. ベースとして元画像を描画
    ctx.drawImage(imageEl, 0, 0);

    // B. 人物だけを型抜きしたテクスチャ
    const personCanvas = document.createElement('canvas');
    personCanvas.width = width;
    personCanvas.height = height;
    const pCtx = personCanvas.getContext('2d');
    if (pCtx) {
      pCtx.drawImage(imageEl, 0, 0);
      pCtx.globalCompositeOperation = 'destination-in';
      pCtx.drawImage(maskCanvas, 0, 0);
    }

    // C. 高輝度（明るい部分）を抽出して発光用画像を作る
    const brightCanvas = document.createElement('canvas');
    brightCanvas.width = width;
    brightCanvas.height = height;
    const bCtx = brightCanvas.getContext('2d');
    if (bCtx) {
      bCtx.drawImage(personCanvas, 0, 0);
      const imgData = bCtx.getImageData(0, 0, width, height);
      const data = imgData.data;
      const thresholdVal = HDR_PARAMS.threshold * 255;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i+1];
        const b = data[i+2];
        const a = data[i+3];
        
        if (a > 0) {
          const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
          if (luminance < thresholdVal) {
            data[i] = 0;
            data[i+1] = 0;
            data[i+2] = 0;
          } else {
            const factor = (luminance - thresholdVal) / (255 - thresholdVal + 1);
            data[i] = Math.min(255, r * (1 + factor * 1.5));
            data[i+1] = Math.min(255, g * (1 + factor * 1.5));
            data[i+2] = Math.min(255, b * (1 + factor * 1.5));
          }
        }
      }
      bCtx.putImageData(imgData, 0, 0);
    }

    // D. 人物自体のトーンアップ (露出・彩度を適用して人物を綺麗に強調)
    const adjustedPersonCanvas = document.createElement('canvas');
    adjustedPersonCanvas.width = width;
    adjustedPersonCanvas.height = height;
    const apCtx = adjustedPersonCanvas.getContext('2d');
    if (apCtx) {
      apCtx.filter = `brightness(${HDR_PARAMS.exposure * 100}%) saturate(${HDR_PARAMS.saturation * 100}%)`;
      apCtx.drawImage(personCanvas, 0, 0);
    }

    // トーンアップした人物を元画像に重ね書き
    ctx.drawImage(adjustedPersonCanvas, 0, 0);

    // E. マルチパス・ブルーム（ぼかしを重ねて自然な発光輪を作る）
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = width;
    glowCanvas.height = height;
    const gCtx = glowCanvas.getContext('2d');

    if (gCtx && HDR_PARAMS.glowRadius > 0) {
      gCtx.globalAlpha = HDR_PARAMS.glowStrength;
      
      // 広範囲の光
      gCtx.filter = `blur(${HDR_PARAMS.glowRadius}px)`;
      gCtx.drawImage(brightCanvas, 0, 0);
      
      // 中範囲の光
      gCtx.filter = `blur(${HDR_PARAMS.glowRadius * 0.4}px)`;
      gCtx.drawImage(brightCanvas, 0, 0);

      // コアの強い光
      gCtx.filter = `blur(${HDR_PARAMS.glowRadius * 0.1}px)`;
      gCtx.drawImage(brightCanvas, 0, 0);
    }

    // 加算（Lighter）で合成
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.drawImage(glowCanvas, 0, 0);
    ctx.restore();

  }, [imageEl, maskCanvas]);

  // ファイルの読み込み
  const loadImage = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target?.result as string;
      setImageSrc(src);

      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        setImageEl(img);
        setAspectRatio(img.naturalWidth / img.naturalHeight);
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
  };

  // ドロップ操作
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadImage(e.dataTransfer.files[0]);
    }
  };

  // ファイル選択
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      loadImage(e.target.files[0]);
    }
  };

  // お手本画像をロード
  const loadDemoImage = () => {
    setImageSrc(DEMO_IMAGE_URL);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImageEl(img);
      setAspectRatio(img.naturalWidth / img.naturalHeight);
    };
    img.src = DEMO_IMAGE_URL;
  };

  // 画像の保存
  const downloadResult = () => {
    if (!canvasRef.current || !imageEl) return;
    
    canvasRef.current.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `hikaru_shashin_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 'image/png');
  };

  // 初期化（元の画面に戻る）
  const resetApp = () => {
    setImageSrc(null);
    setImageEl(null);
    setMaskCanvas(null);
    setAspectRatio(null);
    setShowOriginal(false);
  };

  return (
    <div className="app-container">
      {/* AIの読み込み・準備画面 */}
      {isModelLoading && (
        <div className="loader-overlay">
          <div className="loader-card">
            <div className="spinner"></div>
            <h2>HitoPica を準備しています</h2>
            <p>初回のみ、写真から人を判別するプログラムを準備します。<br />しばらくお待ちください。</p>
            <div className="progress-container">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${modelProgress}%` }}></div>
              </div>
              <span className="progress-text">{modelProgress}% 完了</span>
            </div>
          </div>
        </div>
      )}

      {/* エラー発生時 */}
      {modelError && (
        <div className="error-overlay">
          <div className="error-card">
            <h2>読み込みエラー</h2>
            <p>{modelError}</p>
            <button className="btn btn-primary btn-large" onClick={() => window.location.reload()}>
              もう一度やり直す
            </button>
          </div>
        </div>
      )}

      {/* ヘッダー */}
      <header className="app-header">
        <div className="header-brand">
          <ImageIcon size={28} className="header-icon" />
          <div>
            <h1>HitoPica (ヒトピカ)</h1>
            <p className="header-subtitle">写真の中の「人」を自動で明るく輝かせます</p>
          </div>
        </div>
      </header>

      {/* メイン画面 */}
      <main className="app-main">
        {!imageSrc ? (
          /* 1. 写真選択前の画面 */
          <div className="welcome-section">
            <div 
              className={`upload-card ${isDraggingFile ? 'dragging' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDraggingFile(true); }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="image/*" 
                style={{ display: 'none' }} 
              />
              
              <div className="upload-circle">
                <Upload size={36} className="upload-icon-arrow" />
              </div>
              <h2>ここを押して写真を選ぶ</h2>
              <p>または、ここに写真を引っ張ってきてください</p>
              <span className="upload-tip">※スマホの場合はカメラが起動します</span>
            </div>

            <div className="demo-area">
              <button className="btn btn-secondary btn-large" onClick={loadDemoImage}>
                お手本写真で試してみる
              </button>
            </div>
          </div>
        ) : (
          /* 2. 写真選択後の表示画面 */
          <div className="result-layout">
            
            {/* 画像プレビューエリア */}
            <div className="preview-card">
              <div className="preview-container">
                <div 
                  className="preview-wrapper"
                  style={aspectRatio ? { aspectRatio: `${aspectRatio}`, maxHeight: '60vh', margin: '0 auto' } : undefined}
                  onTouchStart={() => setShowOriginal(true)}
                  onTouchEnd={() => setShowOriginal(false)}
                  onMouseDown={() => setShowOriginal(true)}
                  onMouseUp={() => setShowOriginal(false)}
                  onMouseLeave={() => setShowOriginal(false)}
                >
                  {/* 元の画像 */}
                  <img 
                    src={imageSrc} 
                    alt="元の写真" 
                    className="preview-image base-image"
                    style={{ opacity: (showOriginal || isSegmenting || !maskCanvas) ? 1 : 0 }}
                    draggable={false}
                  />

                  {/* 処理後の画像（Canvas） */}
                  <canvas 
                    ref={canvasRef} 
                    className="preview-image processed-canvas"
                    style={{ opacity: (showOriginal || isSegmenting || !maskCanvas) ? 0 : 1 }}
                  />

                  {/* 処理中のぐるぐる */}
                  {isSegmenting && (
                    <div className="processing-indicator">
                      <RefreshCw className="spin" size={24} style={{ marginBottom: '8px' }} />
                      <span>人物を明るく調整しています...</span>
                    </div>
                  )}

                  {/* 状態ラベル */}
                  <div className="image-badge">
                    {isSegmenting ? '分析中' : showOriginal ? '元の写真' : '光る加工後'}
                  </div>
                </div>
              </div>

              <div className="preview-help">
                <Info size={16} className="help-icon" />
                <span>写真を指で「長押し」している間だけ、元の写真に戻ります。</span>
              </div>
            </div>

            {/* 操作パネル（非常にシンプルなボタン構成） */}
            <div className="action-panel">
              <div className="action-buttons">
                
                <button 
                  className="btn btn-primary btn-large btn-shadow" 
                  onClick={downloadResult}
                  disabled={isSegmenting || !maskCanvas}
                >
                  <Download size={20} style={{ marginRight: '10px' }} />
                  できあがった写真を保存する
                </button>

                <button 
                  className="btn btn-secondary btn-large" 
                  onClick={() => setShowOriginal(!showOriginal)}
                  disabled={isSegmenting || !maskCanvas}
                >
                  <Eye size={20} style={{ marginRight: '10px' }} />
                  {showOriginal ? '光る写真にする' : '元の写真と見比べる'}
                </button>

                <button 
                  className="btn btn-outline btn-large" 
                  onClick={resetApp}
                >
                  <ArrowLeft size={18} style={{ marginRight: '8px' }} />
                  別の写真にする
                </button>
              </div>

              <div className="safety-hint">
                <span>※ 選んだ写真が外部のサーバーに送られることはありません。この端末の中で安全に処理されます。</span>
              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}
