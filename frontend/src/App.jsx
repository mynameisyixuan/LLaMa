import React, { useState, useRef } from 'react';

const API_BASE = 'http://localhost:8000/api'; // 指向你的 FastAPI 后端地址

export default function App() {
  const [activeModule, setActiveModule] = useState('classify'); // 'classify' 或 'grade'
  const [classifyMode, setClassifyMode] = useState('single');   // 'single' 或 'batch'

  // 单字段状态
  const [singleInput, setSingleInput] = useState('');
  const [singleResult, setSingleResult] = useState(null);

  // 批量文件状态
  const [file, setFile] = useState(null);
  const [batchResults, setBatchResults] = useState([]);

  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);

  // 1. 处理单字段分类请求
  const handleSingleSubmit = async () => {
    if (!singleInput.trim()) return;
    setIsLoading(true);
    setSingleResult(null);
    try {
      const response = await fetch(`${API_BASE}/classify/single`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ field_name: singleInput })
      });
      const data = await response.json();
      setSingleResult(data);
    } catch (error) {
      alert("请求失败，请检查后端服务是否已启动");
    } finally {
      setIsLoading(false);
    }
  };

  // 2. 处理批量文件上传请求 (分类或分级)
  const handleBatchSubmit = async () => {
    if (!file) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    const endpoint = activeModule === 'classify' ? '/classify/batch' : '/grade/batch';

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData, // fetch 自动处理 multipart/form-data boundary
      });
      const data = await response.json();
      setBatchResults(data);
    } catch (error) {
      alert("批量处理失败，请检查文件或后端服务");
    } finally {
      setIsLoading(false);
    }
  };

  // 导出 CSV 功能
  const exportToCSV = () => {
    if (batchResults.length === 0) return;
    let headers = [];
    let csvContent = "";

    if (activeModule === 'classify') {
      headers = ['输入字段', '预测大类(Llama)', '预测子类(Llama)'];
      csvContent = batchResults.map(row => `${row.field_name},${row.main_category},${row.sub_category}`).join('\n');
    } else {
      headers = ['输入字段', '数据级别(BERT)'];
      csvContent = batchResults.map(row => `${row.field_name},Level ${row.level}`).join('\n');
    }

    const finalCsv = headers.join(',') + '\n' + csvContent;
    const blob = new Blob(['\uFEFF' + finalCsv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${activeModule === 'classify' ? '分类' : '分级'}结果_${Date.now()}.csv`;
    link.click();
  };

  // 切换模块时清空数据
  const handleModuleSwitch = (module) => {
    setActiveModule(module);
    setFile(null);
    setBatchResults([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-800">
      {/* 顶部 Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-600 p-2 rounded-lg shadow-sm">
              📊
            </div>
            <h1 className="text-xl font-bold tracking-wide text-slate-800">
              数据资产 <span className="text-indigo-600 font-black">AI 引擎</span>
            </h1>
          </div>
          <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200">
            <button
              onClick={() => handleModuleSwitch('classify')}
              className={`px-6 py-2 rounded-md font-medium text-sm transition-all ${activeModule === 'classify' ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
            >
              分类模块 (Llama)
            </button>
            <button
              onClick={() => handleModuleSwitch('grade')}
              className={`px-6 py-2 rounded-md font-medium text-sm transition-all ${activeModule === 'grade' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
            >
              分级模块 (BERT)
            </button>
          </div>
        </div>
      </header>

      {/* 主体内容区 */}
      <main className="max-w-6xl mx-auto px-4 py-8">

        {/* ==================== 分类模块视图 ==================== */}
        {activeModule === 'classify' && (
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 min-h-[500px] animate-in fade-in duration-300">
            <div className="flex justify-between items-end mb-6 border-b border-slate-100 pb-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-800 flex items-center">
                  ⚡ 智能数据分类
                </h2>
                <p className="text-slate-500 mt-2 text-sm">基于 Llama 大语言模型，识别字段数据的业务属性</p>
              </div>
              <div className="flex space-x-3">
                <button onClick={() => {setClassifyMode('single'); setBatchResults([]);}} className={`border px-4 py-2 rounded-lg text-sm font-medium flex items-center transition-colors ${classifyMode === 'single' ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-slate-50 text-slate-500 hover:bg-slate-100'}`}>
                  单字段输入
                </button>
                <button onClick={() => {setClassifyMode('batch'); setSingleResult(null);}} className={`border px-4 py-2 rounded-lg text-sm font-medium flex items-center transition-colors ${classifyMode === 'batch' ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-slate-50 text-slate-500 hover:bg-slate-100'}`}>
                  文件批量分析
                </button>
              </div>
            </div>

            {/* 模式 A：单字段输入 */}
            {classifyMode === 'single' && (
              <div className="max-w-3xl py-4">
                <div className="flex space-x-3 mb-8">
                  <input
                    type="text"
                    value={singleInput}
                    onChange={(e) => setSingleInput(e.target.value)}
                    placeholder="输入需要分类的字段，例如：账号登录密码"
                    className="flex-1 pl-4 pr-4 py-3 border border-slate-300 rounded-lg outline-none shadow-sm focus:ring-2 focus:ring-indigo-500"
                  />
                  <button
                    onClick={handleSingleSubmit}
                    disabled={isLoading || !singleInput}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-lg font-medium flex items-center justify-center shadow-sm disabled:opacity-50"
                  >
                    {isLoading ? '加载中...' : '🔍'}
                    {isLoading ? '推理中...' : '开始分析'}
                  </button>
                </div>
                {singleResult && (
                  <div className="bg-indigo-50/50 border border-indigo-100 rounded-xl p-6 shadow-sm">
                    <h3 className="text-sm font-semibold mb-4 text-slate-600 uppercase tracking-wider">单字段分类结果</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white p-4 rounded-lg border border-slate-100 shadow-sm">
                        <p className="text-sm text-slate-400 mb-1">大类 (Main Category)</p>
                        <p className="font-bold text-indigo-700 text-xl">{singleResult.main_category}</p>
                      </div>
                      <div className="bg-white p-4 rounded-lg border border-slate-100 shadow-sm">
                        <p className="text-sm text-slate-400 mb-1">子类 (Sub Category)</p>
                        <p className="font-bold text-blue-600 text-xl">{singleResult.sub_category}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 模式 B：批量文件输入 */}
            {classifyMode === 'batch' && (
              <BatchUploader
                file={file} setFile={setFile} fileInputRef={fileInputRef}
                isLoading={isLoading} onSubmit={handleBatchSubmit}
                results={batchResults} onExport={exportToCSV} module="classify"
              />
            )}
          </div>
        )}

        {/* ==================== 分级模块视图 (仅批量) ==================== */}
        {activeModule === 'grade' && (
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 min-h-[500px] animate-in fade-in duration-300">
            <div className="flex justify-between items-end mb-6 border-b border-slate-100 pb-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-800 flex items-center">
                  🛡️ 敏感数据分级
                </h2>
                <p className="text-slate-500 mt-2 text-sm">基于 BERT 微调模型，批量计算并划定数据敏感度级别</p>
              </div>
              <div className="bg-emerald-50 text-emerald-700 border border-emerald-200 px-3 py-1.5 rounded-lg text-xs font-medium">
                纯批量文件处理模式
              </div>
            </div>

            <BatchUploader
                file={file} setFile={setFile} fileInputRef={fileInputRef}
                isLoading={isLoading} onSubmit={handleBatchSubmit}
                results={batchResults} onExport={exportToCSV} module="grade"
            />
          </div>
        )}

      </main>
    </div>
  );
}

// 提取出的公共组件：批量上传器与结果表格
function BatchUploader({ file, setFile, fileInputRef, isLoading, onSubmit, results, onExport, module }) {
  const isClassify = module === 'classify';
  const colorBase = isClassify ? 'indigo' : 'emerald';

  return (
    <div className="py-4">
      <div className="flex flex-col sm:flex-row items-center justify-between bg-slate-50 p-4 border border-slate-200 rounded-lg shadow-sm gap-4">
        <div className="flex-1 flex items-center space-x-3 w-full">
          <button onClick={() => fileInputRef.current?.click()} className="flex items-center space-x-2 bg-white border border-slate-300 text-slate-700 hover:bg-slate-100 px-4 py-2.5 rounded-md shadow-sm transition-colors">
            📤
            <span className="font-medium text-sm">选择 TXT / CSV 文件</span>
          </button>
          <input type="file" accept=".csv,.txt" className="hidden" ref={fileInputRef} onChange={(e) => setFile(e.target.files[0])} />
          {file ? (
            <span className={`text-sm font-medium text-${colorBase}-600 flex items-center bg-${colorBase}-50 px-3 py-1.5 rounded-md border border-${colorBase}-100`}>
              ✅ {file.name}
            </span>
          ) : (
            <span className="text-sm text-slate-400">未选择文件</span>
          )}
        </div>
        <button onClick={onSubmit} disabled={!file || isLoading} className={`flex items-center justify-center space-x-2 bg-${colorBase}-600 hover:bg-${colorBase}-700 text-white px-8 py-2.5 rounded-md shadow-sm font-medium disabled:opacity-50 transition-colors text-sm`}>
          {isLoading ? '加载中' : '▶️'}
          <span>{isLoading ? 'AI 批量处理中...' : '开始处理'}</span>
        </button>
      </div>

      {results.length > 0 && (
        <div className="border border-slate-200 rounded-lg overflow-hidden shadow-sm bg-white mt-6 animate-in slide-in-from-bottom-4 duration-500">
          <div className="bg-slate-50 px-5 py-3 border-b border-slate-200 flex justify-between items-center">
            <h3 className="font-bold text-slate-700 flex items-center text-sm">
              📄 处理完成清单 ({results.length} 条)
            </h3>
            <button onClick={onExport} className={`text-xs flex items-center bg-white border border-${colorBase}-200 text-${colorBase}-600 hover:bg-${colorBase}-50 px-3 py-1.5 rounded transition-colors font-medium shadow-sm`}>
              📥 导出 CSV
            </button>
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            <table className="w-full text-left border-collapse">
              <thead className="sticky top-0 bg-slate-100 shadow-sm">
                <tr className="text-slate-600 text-xs uppercase tracking-wider">
                  <th className="p-4 font-medium border-b border-slate-200">输入字段内容</th>
                  {isClassify ? (
                    <>
                      <th className="p-4 font-medium border-b border-slate-200">预测大类</th>
                      <th className="p-4 font-medium border-b border-slate-200">预测子类</th>
                    </>
                  ) : (
                    <th className="p-4 font-medium border-b border-slate-200 text-center">安全定级 (BERT)</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {results.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50">
                    <td className="p-4 text-slate-800 font-medium">{row.field_name}</td>
                    {isClassify ? (
                      <>
                        <td className="p-4 text-indigo-700 font-medium">{row.main_category}</td>
                        <td className="p-4 text-slate-600">{row.sub_category}</td>
                      </>
                    ) : (
                      <td className="p-4 text-center">
                        <span className={`inline-flex items-center justify-center px-4 py-1 rounded-full text-xs font-bold border ${
                          row.level === '1' ? 'bg-green-100 text-green-700 border-green-200' :
                          row.level === '2' ? 'bg-blue-100 text-blue-700 border-blue-200' :
                          row.level === '3' ? 'bg-orange-100 text-orange-700 border-orange-200' :
                          'bg-red-100 text-red-700 border-red-200'
                        }`}>
                          Level {row.level}
                        </span>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}