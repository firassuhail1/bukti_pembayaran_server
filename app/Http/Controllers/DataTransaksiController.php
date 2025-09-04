<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use Illuminate\Http\Request;
use App\Models\DataTransaksi;

class DataTransaksiController extends Controller
{
    public function index()
    {
        $data = DataTransaksi::all();

        return response()->json(['success' => true, 'data' => $data], 200);
    }

    public function store(Request $request)
    {
        // Validasi input
        $validated = $request->validate([
            'nama' => 'required|string|max:255',
            'norek_tujuan' => 'required|string|max:50',
            'nominal' => 'required',
            'payment_method' => 'required|string|max:100',
            'created_at' => 'nullable|date',
        ]);

        // Jika `created_at` tidak dikirim, gunakan waktu saat ini
        $validated['created_at'] = $validated['created_at'] ?? Carbon::now();
        $validated['updated_at'] = $validated['created_at'];

        // Simpan ke database
        DataTransaksi::create($validated);

        return response()->json([
            'success' => true,
            'message' => 'Transaksi berhasil disimpan!'
        ], 200);
    }

    public function edit(Request $request, $id) 
    {
        $created_at = Carbon::parse($request->created_at)->format('Y-m-d H:i:s');

        DataTransaksi::where('id', $id)
            ->update([
                'nama' => $request->nama,
                'norek_tujuan' => $request->norek_tujuan,
                'nominal' => $request->nominal,
                'payment_method' => $request->payment_method,
                'created_at' => $created_at,
            ]);

        return response()->json(['success' => true], 200);
    }

    public function delete(Request $request, $id)
    {
        DataTransaksi::where('id', $id)->delete();

        return response()->json(['success' => true], 200);
    }
}
