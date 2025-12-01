<?php


use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use Symfony\Component\Process\Process;
use App\Http\Controllers\DataTransaksiController;

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::get('/data-transaksi', [DataTransaksiController::class, 'index']);
Route::post('/data-transaksi', [DataTransaksiController::class, 'store']);
Route::put('/data-transaksi/{id}', [DataTransaksiController::class, 'edit']);
Route::delete('/data-transaksi/{id}', [DataTransaksiController::class, 'delete']);

Route::post('/process-image', function (Request $request) {
    if (!$request->hasFile('image')) {
        return response()->json(['error' => 'No image uploaded'], 400);
    }

    $image = $request->file('image');
    $filename = time() . '.' . $image->getClientOriginalExtension();
    $path = storage_path('app/public/' . $filename);
    $image->move(storage_path('app/public'), $filename);

    // Path ke Python dalam Virtual Environment
    $venvPath = base_path('venv');
    $venvPython = base_path('venv/Scripts/python');
    $scriptPath = base_path('scripts\validate_bukti.py');

    // Set Environment Variables (tambahkan SYSTEMROOT)
    $env = [
        'VIRTUAL_ENV' => $venvPath,
        'PYTHONHOME' => '',
        'PYTHONPATH' => $venvPath . '/Lib/site-packages',
        'PATH' => $venvPath . '/Scripts' . PATH_SEPARATOR . getenv('PATH'),
        'SYSTEMROOT' => getenv('SYSTEMROOT'),  // Tambahkan ini
        'WINDIR' => getenv('WINDIR'),  // Tambahkan ini juga
    ];

    // Pastikan Python berjalan dalam virtual environment
    $process = new Process([$venvPython, $scriptPath, $path]);
    $process->setEnv($env);

    $process->run();

    if (!$process->isSuccessful()) {
        // Hapus file jika ada error
        if (file_exists($path)) {
            unlink($path);
        }

        return response()->json([
            'success' => false,
            'error' => 'Python script failed',
            'details' => $process->getErrorOutput()
        ], 500);
    }

    // Hapus gambar setelah diproses
    // if (file_exists($path)) {
    //     unlink($path);
    // }
    
    return response()->json([
        'success' => true,
        'message' => 'Image processed',
        'output' => json_decode($process->getOutput(), true)
    ]);
});