<?php

namespace Database\Seeders;

use Carbon\Carbon;
use App\Models\User;
// use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use App\Models\DataTransaksi;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // User::factory(10)->create();

        User::factory()->create([
            'name' => 'Test User',
            'email' => 'test@example.com',
        ]);

        DataTransaksi::create([
            'nama' => 'Rizal',
            'norek_tujuan' => '0311334726',
            'nominal' => '100000',
            'payment_method' => 'bank',
            'created_at' => Carbon::parse('2024-03-22 01:01:01'), // Custom timestamp
            'updated_at' => Carbon::parse('2024-03-22 01:01:01'),
        ]);

        DataTransaksi::create([
            'nama' => 'Luthfi',
            'norek_tujuan' => '1135679569',
            'nominal' => '2000000',
            'payment_method' => 'bank',
            'created_at' => Carbon::parse('2025-03-21 01:01:01'), // Custom timestamp
            'updated_at' => Carbon::parse('2025-03-21 01:01:01'),
        ]);

        DataTransaksi::create([
            'nama' => 'Rahma',
            'norek_tujuan' => '9000027163774',
            'nominal' => '20300',
            'payment_method' => 'e-wallet',
            'created_at' => Carbon::parse('2021-10-13 01:01:01'), // Custom timestamp
            'updated_at' => Carbon::parse('2021-10-13 01:01:01'),
        ]);

        DataTransaksi::create([
            'nama' => 'Mutiara',
            'norek_tujuan' => '1550005696656',
            'nominal' => '40000',
            'payment_method' => 'bank',
            'created_at' => Carbon::parse('2018-04-30 01:01:01'), // Custom timestamp
            'updated_at' => Carbon::parse('2018-04-30 01:01:01'),
        ]);

        DataTransaksi::create([
            'nama' => 'Ananta Budi',
            'norek_tujuan' => '7415053827',
            'nominal' => '230031',
            'payment_method' => 'bank',
            'created_at' => Carbon::parse('2019-05-23 01:01:01'), // Custom timestamp
            'updated_at' => Carbon::parse('2019-05-23 01:01:01'),
        ]);

        DataTransaksi::create([
            'nama' => 'Chisaria',
            'norek_tujuan' => '126085712558997',
            'nominal' => '323300',
            'payment_method' => 'bank',
            'created_at' => Carbon::parse('2025-03-21 01:01:01'), // Custom timestamp
            'updated_at' => Carbon::parse('2025-03-21 01:01:01'),
        ]);
    }
}
